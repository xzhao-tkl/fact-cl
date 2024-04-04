import os
import ast
from typing import Iterator, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from mask_dataset import MaskedDataset
from modules.bert_base_model import BERTBaseModel
from modules.xlmr_base_model import XLMBaseModel
from prober import Prober
from utils import batchify, get_logger
from constants import INTERVENTION_ROOT

logger = get_logger("neuron_intervention.log", __name__)

class NeuronIntervener:
    def __init__(self, 
                 dataset: MaskedDataset, 
                 neuron_type='acts',
                 neuron_operator='top10_1',
                 device='cuda:0', 
                 batch_size=256,
                 target_rels: Optional[list[str]]=None,
                 base_langs: Optional[list[str]]=None, 
                 test_langs: Optional[list[str]]=None) -> None:
        
        self.dataset = dataset
        self.neuron_type = neuron_type
        self.neuron_operator = neuron_operator
        self.batch_size = int(batch_size)
        if not neuron_operator.startswith("top"):
            raise ValueError(f"Unsupported active neuron operation method - {neuron_operator}")
        
        if "_" not in neuron_operator: 
            raise ValueError(f"`_` must be set in neuron_operator to set the IDR (incrase or decrase rate), but get {neuron_operator}")
        
        self.rate = float(neuron_operator.split('_')[1])
        self.k = int(neuron_operator.split('_')[0][3:])
        if self.dataset.model_name == 'xlmr':
            self.model = XLMBaseModel(device=torch.device(device), collect_mode=True)
            self.neuron_probing_model = XLMBaseModel(device=torch.device(device), intervened=True, intervened_neuron_type=self.neuron_type, collect_mode=False)
        elif self.dataset.model_name == 'mbert':
            self.model = BERTBaseModel(device=torch.device(device), collect_mode=True)
            self.neuron_probing_model = BERTBaseModel(device=torch.device(device), intervened=True, intervened_neuron_type=self.neuron_type, collect_mode=False)
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet")
        
        self.model_name = self.neuron_probing_model.name
        self.prober = Prober(dataset, match_type='full-match', model=self.model)
        self.tokenized_objs = dataset.get_all_tokenized_objs()

        self.base_langs = base_langs if base_langs else self.dataset.langs
        self.test_langs = test_langs if test_langs else self.dataset.langs
        self.target_rels = target_rels if test_langs else self.dataset.rels

        print(f"device: {device}")
        print(f"batch_size: {batch_size}")
        print(f"base_langs: {base_langs}")
        print(f"test_langs: {test_langs}")
        print(f"target_rels: {target_rels}")
        print(f"neuron_type: {neuron_type}")
        print(f"neuron_operator: {neuron_operator}")

    def generate_neuron_mask(self, summation: np.ndarray) -> np.ndarray:
        summation = summation.reshape(self.model.layer_num, -1)
        topk = np.argpartition(summation.ravel(), -self.k)[-self.k:]
        indices = np.array(np.unravel_index(topk, summation.shape)).T
        mask = np.zeros(summation.shape)
        mask[indices[:, 0], indices[:, 1]] = 1
        return (1 - mask) + mask * self.rate

    def generate_crosslingual_batch_sentences(self, rel: str, base_lang, iterate_type, obj: Optional[str] = None, reload: bool = False) -> Iterator:
        """
        Args:
            rel (str): relation URI
            base_lang (str): for checking dumped file
            iterate_type (str): for checking dumped file
            obj (_type_, optional): If obj is None, generate batch for all sentences by lang-rel. Otherwise, generate batch by lang-rel-obj
        Yields:
            lang, batch_uuids, batch_sents
        """
        for test_lang in self.test_langs:
            if len(list(self.tokenized_objs[test_lang][rel].keys())) == 0:
                print(f"Skipped {test_lang, rel} in generate_crosslingual_batch_sentences")
                continue
            if not self.is_exist_probing_result_dump(
                reload = reload,
                dump_file_path = self.get_dump_fn(rel, base_lang, test_lang, obj, iterate_type)):
                continue
                        
            all_sents = []
            all_uuids = []
            maxlen = max(list(self.tokenized_objs[test_lang][rel].keys()))
            for i in range(maxlen):
                df = self.dataset.get_lang_type(test_lang, rel)
                if obj is not None:
                    df = df[df['obj_uri']==obj]
                all_sents.extend(self.dataset.replace_with_mask(df["sent"].tolist(), i + 1))
                all_uuids.extend(df["uuid"].tolist())

            for batch in batchify(list(zip(all_uuids, all_sents)), self.batch_size):
                batch_uuids = [elem[0] for elem in batch]
                batch_sents = [elem[1] for elem in batch]
                yield test_lang, batch_uuids, batch_sents

    def generate_crosslingual_batch_sentences_by_obj(self, rel: str, obj: str, base_lang: str, iterate_type: str) -> Iterator:
        return self.generate_crosslingual_batch_sentences(rel, base_lang, iterate_type, obj)

    def neuron_intervention_iterator(self, iterate_type='rel', reload=False) -> Iterator[tuple[str, str, str, Optional[str], list, list, np.ndarray, Optional[torch.Tensor]]]:
        """
        Returns: iterator of a two-element tuple
            - base_lang: base language used for dectect active neurons. 
            - test_lang: test language of the batch. It is used for later evaluation.
            - batch_uuids: batch of uuids, which is used for p1 score evaluation by matching it with golden data
            - sentences: batch of sentences for feed-forward processing by ML-LMs
            - active_neurons_per_layer (torch.tensor): layer_num x layer_size, the map of active neurons used for intervention
        """
        assert self.target_rels != None
        for rel in self.target_rels:
            lang2probing = self.prober.probe_objs_per_lang(rel=rel, probing_type=self.neuron_type, reload=False)[1]
            print(f"\nPerforming neuron invervention by relation {rel} - {self.dataset.display_rel(rel)}")
            for base_lang in self.dataset.get_langs_in_rel(rel, cand_langs=self.base_langs):
                _, _, overall_summation, obj2summation = lang2probing[base_lang]
                
                if iterate_type not in ['rel', 'obj']:
                    raise ValueError(f"Unsupported iteration type: {iterate_type}. It must be either `rel` or `obj`")
                
                if iterate_type == 'rel' and overall_summation is not None:
                    neuron_masks = self.generate_neuron_mask(overall_summation)
                    for test_lang, batch_uuids, batch_sents in self.generate_crosslingual_batch_sentences(rel, base_lang, iterate_type):
                        yield rel, base_lang, test_lang, None, batch_uuids, batch_sents, neuron_masks, None
                elif iterate_type == 'obj' and obj2summation is not None:
                    obj2neuron_masks = {}
                    for obj in obj2summation.keys():
                        for test_lang, batch_uuids, batch_sents in self.generate_crosslingual_batch_sentences_by_obj(rel, obj, base_lang, iterate_type):
                            # embedding = self.collect_base_lang_embeddings(base_lang, obj, matches_by_rel).type(torch.float32).to(self.neuron_probing_model.device)
                            if obj not in obj2neuron_masks:
                                obj2neuron_masks[obj] = self.generate_neuron_mask(obj2summation[obj])
                            yield rel, base_lang, test_lang, obj, batch_uuids, batch_sents, obj2neuron_masks[obj], None

    def collect_base_lang_embeddings(self, base_lang, obj, correct_matches_by_rel):
        embeddings = []
        base_lang_sents = self.prober.sent_iter.get_matches_by_obj_lang(obj_uri=obj, lang=base_lang, correct_matches=correct_matches_by_rel)
        for info in self.model.collect_neurons(base_lang_sents, reload=False):
            embeddings.append(info[self.neuron_type].mean(dim=1))
        return torch.stack(embeddings).mean(dim=0)

    def probing_with_intervention(self, iterate_type='obj', reload: bool = False):
        uuid_info = self.dataset.get_uuid_info_all_lang()
        frame = pd.DataFrame(columns=["base_lang", "test_lang", "uuid", "obj", "sub", "obj_uri", "sub_uri", "sent", "mask_num", "prediction", "pred_ids"])
        prev_batch = None 
        for rel, base_lang, test_lang, obj, batch_uuids, batch_sents, neuron_masks, obj_embedding in \
            tqdm(self.neuron_intervention_iterator(iterate_type=iterate_type, reload=reload), desc=f"Probing factual knowledge based on intervened ML-LMs"):
            # print(rel, base_lang, test_lang, obj, len(batch_uuids))
            if prev_batch is None:
                prev_batch = (rel, base_lang, test_lang, obj)
            
            if prev_batch != (rel, base_lang, test_lang, obj):
                self.dump_probing_result(
                    frame = frame,
                    dump_file_path = self.get_dump_fn(prev_batch[0], prev_batch[1], prev_batch[2], prev_batch[3], iterate_type),
                    logging_info = f"The previous batch is {prev_batch}, the current batch is {(rel, base_lang, test_lang, obj)}")
                frame = pd.DataFrame(columns=["base_lang", "test_lang", "uuid", "obj", "sub", "obj_uri", "sub_uri", "sent", "mask_num", "prediction", "pred_ids"])
                prev_batch = (rel, base_lang, test_lang, obj)
            
            self.neuron_probing_model.inject_intervention_resources(
                neuron_masks = torch.from_numpy(neuron_masks).type(torch.float32).to(self.model.device),
                obj_embeddings = obj_embedding)
            self.neuron_probing_model.inject_mask_indices(self.get_mask_indices(batch_sents))
            mask_tokens_ls, mask_tokenids_ls = self.neuron_probing_model.get_mask_tokens_ids(batch_sents)
            item = {
                "base_lang": base_lang,
                "test_lang": test_lang, 
                "uuid": batch_uuids,
                "obj":      [uuid_info[uuid][test_lang]['obj'] for uuid in batch_uuids],
                "sub":      [uuid_info[uuid][test_lang]['sub'] for uuid in batch_uuids],
                "obj_uri":  [uuid_info[uuid][test_lang]['obj_uri'] for uuid in batch_uuids],
                "sub_uri":  [uuid_info[uuid][test_lang]['sub_uri'] for uuid in batch_uuids],
                "sent": batch_sents,
                "mask_num": [len(tokens) for tokens in mask_tokens_ls],
                "prediction": mask_tokens_ls,
                "pred_ids": mask_tokenids_ls}
            frame = pd.concat([frame, pd.DataFrame(item)])
        
        if prev_batch:
            self.dump_probing_result(
                frame = frame,
                dump_file_path = self.get_dump_fn(prev_batch[0], prev_batch[1], prev_batch[2], prev_batch[3], iterate_type))
        else:
            logger.info("The prev_batch is None")
            # raise ValueError("The prev_batch after iteration shouldn't be None")
    
    def get_dump_fn(self, test_rel, base_lang, test_lang, obj, iterate_type):
        if obj and iterate_type == "rel":
            raise ValueError("When iterate_type equals to `rel`, the obj should be None all the time")
        elif not obj and iterate_type == "obj":
            raise ValueError("When iterate_type equals to `obj`, the obj cannot be None all the time")
        
        path = os.path.join(
            INTERVENTION_ROOT[self.model_name] + f"-{self.neuron_type}",
            f'{iterate_type}-{self.neuron_operator}', 
            test_rel, 
            f"{base_lang}-base", 
            f"{test_lang}-test")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{obj}.csv")
    
    def is_exist_probing_result_dump(self, dump_file_path, reload):
        if os.path.exists(dump_file_path) and reload==False:
            return False
        return True
    
    def dump_probing_result(self, dump_file_path, frame: pd.DataFrame, logging_info: Optional[str] = None):
        if dump_file_path == None:
            logger.info(f"{dump_file_path} is already generated.")
        else:
            logger.info(f"Dumped new frame to {dump_file_path}. Size: {len(frame)}.")
            frame.to_csv(dump_file_path)
        if logging_info is not None:
            logger.info(logging_info)

    def get_mask_indices(self, batch_sents): 
        encoded_inputs = self.neuron_probing_model.tokenizer(batch_sents, padding=True, return_tensors="pt").to(self.neuron_probing_model.device)
        return (encoded_inputs.input_ids == self.neuron_probing_model.tokenizer.mask_token_id).type(torch.float32).to(self.model.device)

if __name__ == "__main__":
    import argparse
    from mask_dataset import MaskedDataset
    
    dataset = MaskedDataset()
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--neuron_type', default='acts')
    parser.add_argument('--iterate_type', default='obj')
    
    parser.add_argument('--neuron_operator', nargs='+', default=['top30_1.8'])
    parser.add_argument('--base_langs', nargs='+', default=['en'])
    parser.add_argument('--test_langs', nargs='+', default=['en', 'es', 'tr'])
    parser.add_argument('--target_rels', nargs='+', default=['P103'])

    parser.add_argument('--all_base_langs', default=False)
    parser.add_argument('--all_test_langs', default=False)
    parser.add_argument('--all_rels', default=False)
    
    args = parser.parse_args()

    base_langs = ['en', 'id', 'pl', 'zh']
    test_langs = [
        'en', 'es', 'de', 'da', 
        'id', 'ms', 'vi', 'sv', 
        'pl', 'hu', 'sk', 'cs', 
        'zh', 'ko', 'sr', 'ja']
    test_langs = ["it", "nl", "pt", "ca", "tr", "fr", "af", "ro", "gl", "fa", "el", "cy"]
    target_rels = dataset.rels if args.all_rels else args.target_rels
    base_langs = base_langs if args.all_base_langs else args.base_langs
    test_langs = test_langs if args.all_test_langs else args.test_langs

    for operator in args.neuron_operator:
        neuronItvr = NeuronIntervener(
            dataset=dataset,     
            batch_size=args.batch_size,

            base_langs=base_langs, 
            test_langs=test_langs, 
            target_rels=target_rels,             

            neuron_operator=operator,
            neuron_type=args.neuron_type, 
            device=args.device)
        neuronItvr.probing_with_intervention(
            iterate_type= args.iterate_type,
            reload=False)