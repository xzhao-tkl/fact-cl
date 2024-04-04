import os
import json
import torch
import pandas as pd
from ResourcePool import ResourcePool
from evaluation import read_pred
from utils import load_objects
from mask_dataset import MaskedDataset
from modules.xlmr_base_model import XLMBaseModel
from modules.bert_base_model import BERTBaseModel

XLMR_PRED_ROOT = "../result/prediction-xlmr"
MBERT_PRED_ROOT = "../result/prediction-mbert"

PRED_ROOT = {
    "xlmr": XLMR_PRED_ROOT, 
    "mbert": MBERT_PRED_ROOT
}
def parse_list(pred):
    return ast.literal_eval(pred)

def tokens2id(pred, tokenizer):
    token_ids = []
    for tokens in pred:
        assert(all([len(token) == 1 and type(token[0]) == str for token in tokens]) )
        tokens = [token[0] for token in tokens]
        token_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    return token_ids

def batchify(sents, batch_size):
    l = len(sents)
    for ndx in range(0, l, batch_size):
        yield sents[ndx:min(ndx + batch_size, l)]
        
def check_pred_exists(root, lang, rel, objs, overwrite):
    folder = os.path.join(root, lang)
    if not os.path.exists(folder):
        os.mkdir(folder)
    result_fp = os.path.join(root, lang, '{}-{}.csv'.format(lang, rel))
    if overwrite:
        return result_fp
    if os.path.exists(result_fp):
        # print("{} has been already created.".format(result_fp))
        return None
    
    # print('Start to predict masked tokens for lang-{}, relation {}'.format(lang, rel))
    if len(list(objs[rel].keys())) == 0:
        print("lang-{}, relation {} has zero objects.".format(lang, rel))
        return None
    
    return result_fp
    
def predict_mask_tokens(model, dataset, objs, lang, rel, root, overwrite=False):
    result_fp = check_pred_exists(root, lang, rel, objs, overwrite)
    if result_fp is None:
        return
    
    print("Predict mask tokens for {}-{}".format(lang, rel))
    frame = pd.DataFrame(columns=['id', 'sent', 'mask_num', 'prediction'])
    maxlen = max(list(objs[rel].keys()))
    for i in range(maxlen):
        relations = dataset.get_lang_type(lang, rel)
        org_sents = relations['sent']
        ids = relations.index
        sents = dataset.replace_with_mask(org_sents, i+1, model.mask_token)
        batches = batchify(list(zip(ids, sents)), 128)
        for batch in batches:
            ids = list(zip(*batch))[0]
            sents = list(zip(*batch))[1]
            results = model.get_mask_tokens(sents, i+1)
            samples = list(zip(ids, sents, results))
            item = {
                'id': ids,
                'sent': sents,
                'mask_num': i+1,
                'prediction': results, 
                'pred_ids': tokens2id(results, model.tokenizer)
            }
            frame = pd.concat([frame, pd.DataFrame(item)])
    frame.to_csv(result_fp)

    
class MLLMPool(ResourcePool):
    def task(self, args):
        idx, model = self.assign()
        lang, rel, objs = args
        predict_mask_tokens(model, mlama, objs, lang, rel, PRED_ROOT[model.name])
        # print("Thread finish running prediction task for {}-{}, with {}-th {} model".format(lang, rel, idx, model.name))
        self.release(idx)

def predict_all_parallel(dataset, Model, thread_num):
    resources = [Model(torch.device("cuda:{}".format(idx))) for idx in list(range(thread_num))]
    lm_pool = MLLMPool(resources, thread_num)
    print("Initalized {} MBert models and pool for running mask-prediction tasks".format(thread_num))
    
    lang2objs = {}
    for lang in dataset.langs:
        objs = load_objects(lang, resources[0].name, resources[0])
        lang2objs.update({lang:objs})
    print("Initalized lang2objs")
          
    def res_iterator():
        for lang in mlama.langs:
            for rel in lang2objs[lang].keys():
                yield lang, rel, lang2objs[lang]
    
    # print("Start to submit threads")
    lm_pool.run(res_iterator())    
    # print("Finished the mask-prediction tasks")
    
if __name__ == '__main__':
    # Initialization
    device = torch.device("cuda:0")
    model = XLMBaseModel(device)
    # model = BERTBaseModel(device)
    mlama = MaskedDataset("mlama", model.name)
    
    ## Acquire the prediction results for multiple masks in the single thread
    # for lang in mlama.langs:
    #     objs = load_objects(lang, xlmr)
    #     for rel in objs.keys():
    #         predict_mask_tokens_2(xlmr, mlama, objs, lang, rel)
    
    
    # lang = 'af'
    # objs = load_objects(lang, model.name, None)
    # for rel in objs.keys():
    #     predict_mask_tokens(xlmr, mlama, objs, lang, rel, PRED_ROOT[model.name])
    
    lang = "af"
    objs = load_objects(lang, model.name, None)
    predict_mask_tokens(model, mlama, objs, lang, 'P449', PRED_ROOT[model.name], True)
    
#     lang = "da"
#     objs = load_objects(lang, model.name, None)
#     predict_mask_tokens(model, mlama, objs, lang, 'P39', PRED_ROOT[model.name], True)
    
    
    ## Acquire the prediction results for multiple masks in parallel 
    # mlama = MaskedDataset("mlama", "[MASK]", "mbert")
    # predict_all_parallel(mlama, BERTBaseModel, 8)
    