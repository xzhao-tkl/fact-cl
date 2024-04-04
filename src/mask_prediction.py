import ast
from asyncio import as_completed
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from constants import PREDICTION_ROOT
from mask_dataset import MaskedDataset
from modules.abstract_model import AbstractModel
from utils import batchify, split_list
from tqdm import tqdm

def parse_list(pred):
    return ast.literal_eval(pred)


def tokens2id(pred, tokenizer):
    token_ids = []
    for tokens in pred:
        assert all(len(token) == 1 and type(token[0]) == str for token in tokens)
        tokens = [token[0] for token in tokens]
        token_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    return token_ids

def check_pred_exists(root, lang, rel, objs, overwrite):
    folder = os.path.join(root, lang)
    if not os.path.exists(folder):
        os.mkdir(folder)
    result_fp = os.path.join(root, lang, f"{lang}-{rel}.csv")
    if overwrite:
        return result_fp
    if os.path.exists(result_fp):
        return None

    if not list(objs[rel].keys()):
        print(f"lang-{lang}, relation {rel} has zero objects.")
        return None

    return result_fp

def predict_mask_tokens(model: AbstractModel, dataset, objs, lang, rel, root, overwrite=False):
    result_fp = check_pred_exists(root, lang, rel, objs, overwrite)
    if result_fp is None:
        return False

    frame = pd.DataFrame(columns=["id", "sent", "mask_num", "prediction"])
    maxlen = max(list(objs[rel].keys()))
    for i in tqdm(range(maxlen), desc=f"Predict mask tokens for {lang}-{rel}"):
        relations = dataset.get_lang_type(lang, rel)
        org_sents = relations["sent"]
        uuids = relations["uuid"]
        obj_tokens = relations["obj"]
        sub_tokens = relations["sub"]
        obj_uris = relations["obj_uri"]
        sub_uris = relations["sub_uri"]
        ids = relations.index
        sents = dataset.replace_with_mask(org_sents, i + 1)
        batches = batchify(list(zip(ids, sents, uuids, obj_tokens, sub_tokens, obj_uris, sub_uris)), 128)
        for batch in batches:
            batch_ids = [elem[0] for elem in batch]
            batch_sents = [elem[1] for elem in batch]
            batch_uuids = [elem[2] for elem in batch]
            batch_obj_tokens = [elem[3] for elem in batch]
            batch_sub_tokens = [elem[4] for elem in batch]
            batch_obj_uris = [elem[5] for elem in batch]
            batch_sub_uris = [elem[6] for elem in batch]
            
            mask_tokens_ls, mask_tokenids_ls = model.get_mask_tokens_ids(batch_sents)
            item = {
                "id": batch_ids,
                "sent": batch_sents,
                "uuid": batch_uuids,
                "obj": batch_obj_tokens,
                "sub": batch_sub_tokens,
                "obj_uri": batch_obj_uris,
                "sub_uri": batch_sub_uris,
                "mask_num": i + 1,
                "prediction": mask_tokens_ls,
                "pred_ids": mask_tokenids_ls,
            }
            frame = pd.concat([frame, pd.DataFrame(item)])
    frame.to_csv(result_fp)
    return True

def predict_mask_tokens_in_loop(model, dataset, params):
    for param in params:
        objs, lang, rel = param
        predict_mask_tokens(model=model, dataset=dataset, objs=objs, lang=lang, rel=rel, root=PREDICTION_ROOT[model.name])

def predict_all_parallel(dataset: MaskedDataset, Model, thread_num):
    import torch.multiprocessing as multiprocessing
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    
    tokenized_obj = dataset.get_all_tokenized_objs()
    def res_iterator():
        for lang in dataset.langs:
            for rel in tokenized_obj[lang].keys():
                yield tokenized_obj[lang], lang, rel

    params_lists = split_list(list(res_iterator()), thread_num)
    models = [
        Model(torch.device("cuda:{}".format(idx)), collect_mode=False) for idx in list(range(thread_num))
    ]
    
    with ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = []
        for i, params in enumerate(params_lists):
            futures.append(executor.submit(predict_mask_tokens_in_loop, models[i], dataset, params))            

if __name__ == "__main__":
    from modules.bert_base_model import BERTBaseModel
    from modules.xlmr_base_model import XLMBaseModel

    # Initialization
    # device = torch.device("cuda:0")
    # model = BERTBaseModel(device)
    # dataset = MaskedDataset()
    
    # ## Acquire the prediction results for multiple masks in the single thread
    # for lang in ['de'] + dataset.langs:
    #     objs = load_objects(lang=lang, dataset=dataset)
    #     for rel in objs.keys():
    #         predict_mask_tokens(model=model, objs=objs, lang=lang, rel=rel, root=PREDICTION_ROOT[model.name], dataset=dataset)
    
    # main()

    ## Acquire the prediction results for multiple masks in parallel
    dataset = MaskedDataset(model_name='xlmr')
    predict_all_parallel(dataset, XLMBaseModel, 8)
