import ast
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from constants import MODELS, PROBER_ROOT_ON_DISK
from mask_dataset import MaskedDataset
from modules.bert_base_model import BERTBaseModel
from modules.xlmr_base_model import XLMBaseModel
from modules.probeless import get_neuron_ordering_for_all_tags
from sents_iterator import SentIterator
from utils import loader, prober_loader

def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def get_overall_differ(differ_per_tag: dict):
    size = len(differ_per_tag)
    differs = list(differ_per_tag.values())
    overall_differ = np.zeros_like(differs[0])
    for differ in overall_differ:
        overall_differ += differ / (size - 1)
    return overall_differ

def get_uuids_candidates_per_lang(dataset: MaskedDataset, cand_size: int):
    import random
    from pred_evaluation import get_gold_matrix_per_uuid, get_full_match_matrix_by_uuid
    
    gold_langs, gold_uuids, gold_matrix = get_gold_matrix_per_uuid(dataset)
    full_langs, full_uuids, full_matrix = get_full_match_matrix_by_uuid(dataset)
    assert gold_langs == full_langs
    assert gold_uuids == full_uuids
    
    cand_idx = np.where((gold_matrix.sum(axis=0)>=25) & (full_matrix.sum(axis=0)>=5))[0]
    cand_uuids = [gold_uuids[idx] for idx in cand_idx]
    gold_cnt = gold_matrix.sum(axis=0)[cand_idx]
    full_cnt = full_matrix.sum(axis=0)[cand_idx]
    ranked_idx = np.argsort(full_cnt/gold_cnt)[::-1]
    ranked_uuids = [cand_uuids[int(idx)] for idx in ranked_idx]

    lang2uuids = {}
    for lang in dataset.langs:
        matched_idx_by_lang = np.where(full_matrix[full_langs.index(lang)] > 0)[0]
        if len(matched_idx_by_lang) < cand_size:
            uuids_by_lang = [full_uuids[int(idx)] for idx in matched_idx_by_lang]
        else:
            shared_uuids = set(cand_idx.tolist()).intersection(set(matched_idx_by_lang.tolist()))
            if len(shared_uuids) > cand_size:
                uuids_by_lang = list(random.choices(list(shared_uuids), k=cand_size))
            else:
                uuids_by_lang = list(shared_uuids)
                uuids_by_lang.extend(random.choices(list(set(matched_idx_by_lang.tolist()) - set(cand_idx.tolist())), k=cand_size-len(uuids_by_lang)))
            uuids_by_lang = [full_uuids[idx] for idx in uuids_by_lang]
        lang2uuids[lang] = uuids_by_lang
    return lang2uuids, ranked_uuids


class Prober():
    def __init__(self, dataset: MaskedDataset, match_type='full-match', device=None, model=None, batchsize=32) -> None:
        if device is None:
            device = 'cuda:0'
        
        self.dataset = dataset
        self.match_type = match_type
        self.sent_iter = SentIterator(dataset)
        self.model_name = dataset.model_name
        self.batchsize = batchsize
        
        assert self.model_name in MODELS

        if not model:
            device = torch.device(device)
            if self.model_name == 'xlmr':
                self.model = XLMBaseModel(device)
            elif self.model_name == 'mbert':
                self.model = BERTBaseModel(device)
            else:
                raise NotImplementedError(f"Model {self.model_name} is not implemented yet")
        else:
            self.model = model
            self.model_name = model.name
        
    def collect_neurons(self, sents, reload=False):
        return self.model.collect_neurons(sents, reload=False)

    def get_probing_method(self, by):
        defined_probing_methods = ["probeless"]
        try:
            assert by in defined_probing_methods
        except Exception as e:
            raise NotImplementedError(
                f"The probing method {by} is not defined yet. Please use methods from {defined_probing_methods}"
            ) from e

        return getattr(self, by)
    
    @prober_loader
    def probe_objs(self, rel: str, by='probeless', probing_type='acts', reload=False):
        """Probe facutal neurons for different objects (with the same relation), mixing sentences in different languages together
        Perform comparison on objects

        Args:
            rel (str): The rel uri
            by (str): Probing methods. Default by probeless
            probing_type (str): must be one of ['acts', 'outs'], representing for activation and intermidate embedding layer in Transformer

        Returns:
            probing_result (tuple): {
                overall_ranking : list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
                ranking_per_tag : Dictionary with top neurons for every class, with the class name as the key and list of neurons as the values.
                overall_differ
                differ_per_tag
            }
        """
        assert probing_type in ["acts", "outs"]
        probing_method = self.get_probing_method(by)
        correct_matches = self.sent_iter.get_correct_matches(rel=rel, match_type=self.match_type)
        obj2sents = {
            obj_uri: self.sent_iter.get_matches_by_obj(obj_uri, correct_matches)
            for obj_uri in tqdm(self.dataset.get_objs_in_rel(rel), desc=f"Probing neurons across languages for relation {rel}")
        }
        return probing_method(obj2sents, probing_type=probing_type)

    @prober_loader
    def probe_langs(self, rel: str, by='probeless', probing_type='acts', reload=False):
        """Probe facutal neurons for different languages (with the same relation), mixing sentences with different objects together
        Perform comparison on languages

        Args:
            rel (str): The rel uri
            by (str): Probing methods. Default by probeless
            probing_type (str): must be one of ['acts', 'outs'], representing for activation and intermidate embedding layer in Transformer

        Returns:
            probing_result (tuple): {
                overall_ranking : list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
                ranking_per_tag : Dictionary with top neurons for every class, with the class name as the key and list of neurons as the values.
                overall_differ
                differ_per_tag
            }
        """
        assert probing_type in ["acts", "outs"]
        probing_method = self.get_probing_method(by)
        correct_matches = self.sent_iter.get_correct_matches(rel=rel, match_type=self.match_type)
        lang2sents = {
            lang: self.sent_iter.get_matches_by_lang(lang, correct_matches)
            for lang in tqdm(self.dataset.langs, desc=f"Probing neurons across languages for relation {rel}")
        }
        return probing_method(lang2sents, probing_type=probing_type)

    @prober_loader
    def probe_objs_per_lang(self, rel: str, by='probeless', probing_type='acts', reload=False):
        """Probe facutal neurons for the same relation, run for each object
        Perform comparison on objects

        Args:
            rel (str): The rel uri
            lang (str): languages for dis
            by (str): Probing methods. Default by probeless

        Returns:
            obj2matchedlangs (dict): {"Q150": [en, zh, fr, ...], "Q1530": [...], ...}
            rankings (dict): {
                lang: [
                    overall_ranking : list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
                    ranking_per_tag : Dictionary with top neurons for every class, with the class name as the key and list of neurons as the values.
                    differ_per_tag
                ]
            }
        """
        assert probing_type in ["acts", "outs"]
        lang2obj2sents = {}
        probing_result = {}
        obj2matchedlangs = {}
        probing_method = self.get_probing_method(by)
        correct_matches = self.sent_iter.get_correct_matches(rel=rel, match_type=self.match_type)
        for lang in tqdm(self.dataset.langs, desc=f"Probing neurons across objects per lang for relation {rel}"):
            lang2obj2sents[lang] = {}
            for obj_uri in self.dataset.get_objs_in_rel(rel):
                lang2obj2sents[lang][obj_uri] = self.sent_iter.get_matches_by_obj_lang(obj_uri, lang, correct_matches)
                if lang2obj2sents[lang][obj_uri]:
                    if obj_uri in obj2matchedlangs:
                        obj2matchedlangs[obj_uri].add(lang)
                    else:
                        obj2matchedlangs[obj_uri] = set()
            probing_result[lang] = probing_method(lang2obj2sents[lang], probing_type=probing_type)
        obj2matchedlangs = {k: sorted(list(v)) for k, v in obj2matchedlangs.items()}
        return obj2matchedlangs, probing_result

    @prober_loader
    def probe_langs_per_obj(self, rel: str, by='probeless', probing_type='acts', reload=False):
        """Probe facutal neurons for the same relation, run for each object. Comparison are done across languages
        Perform comparison on objects

        Args:
            rel (str): The rel uri
            lang (str): languages for dis
            by (str): Probing methods. Default by probeless

        Returns:
            obj2matchedlangs (dict): {"Q150": [en, zh, fr, ...], "Q1530": [...], ...}
            rankings (dict): {
                obj_uri: [
                    overall_ranking : list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
                    ranking_per_tag : Dictionary with top neurons for every class, with the class name as the key and list of neurons as the values.
                    differ_per_tag
                ]
            }
        """
        assert probing_type in ["acts", "outs"]
        obj2lang2sents = {}
        probing_result = {}
        lang2matchedobjs = {}
        probing_method = self.get_probing_method(by)
        correct_matches = self.sent_iter.get_correct_matches(rel=rel, match_type=self.match_type)
        for obj_uri in tqdm(self.dataset.get_objs_in_rel(rel), desc=f"Probing neurons across languages per object for relation {rel}"):
            obj2lang2sents[obj_uri] = {}
            for lang in self.dataset.langs:
                obj2lang2sents[obj_uri][lang] = self.sent_iter.get_matches_by_obj_lang(obj_uri, lang, correct_matches)
                if obj2lang2sents[obj_uri][lang]:
                    if lang in lang2matchedobjs:
                        lang2matchedobjs[lang].add(obj_uri)
                    else:
                        lang2matchedobjs[lang] = set()
            probing_result[obj_uri] = probing_method(obj2lang2sents[obj_uri], probing_type=probing_type)
        lang2matchedobjs = {k: sorted(list(v)) for k, v in lang2matchedobjs.items()}
        return lang2matchedobjs, probing_result

    def probe_uuid_by_lang_rel(self, lang: str, rel: str, cand_size=500, by='probeless', probing_type='acts', reload=False):
        """Probe facutal neurons for the uuid, run for each uuid. The comparison is done within each language. 
        Perform comparison on objects

        Args:
            rel (str): The rel uri
            lang (str): languages for dis
            by (str): Probing methods. Default by probeless

        Returns:
            uuid2matchedlangs (dict): {"004bca2b-2e00-4872-a644-06eb29a10f55": [en, zh, fr, ...], "004bca2b-2e00-4872-a644-06eb29aasdsf5a5": [...], ...}
            rankings (dict): {
                lang: [
                    overall_ranking : list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
                    ranking_per_tag : Dictionary with top neurons for every class, with the class name as the key and list of neurons as the values.
                    differ_per_tag
                ]
            }
        """
        root_path = os.path.join(PROBER_ROOT_ON_DISK, self.dataset.model_name, 'probe_uuids_per_lang_rel')
        os.makedirs(root_path, exist_ok=True)
        dump_fn = os.path.join(root_path, f'{lang}_{rel}_{cand_size}_{probing_type}_lang2uuid2sents.pkl')
        if os.path.exists(dump_fn) and reload==False:
            with open(dump_fn, 'rb') as fp:
                return pickle.load(fp)
        
        assert probing_type in ["acts", "outs"]
        uuid2sents = {}
        probing_result = {}
        probing_method = self.get_probing_method(by)
        correct_matches = self.sent_iter.get_correct_matches(rel=rel, match_type=self.match_type)
        
        lang2uuids, ranked_uuids = get_uuids_candidates_per_lang(self.dataset, 10000)
        cnt = 0
        for rel_uri in correct_matches:
            for uuid in correct_matches[rel_uri]:
                if cnt > cand_size:
                    break
                if uuid in lang2uuids[lang] and lang in correct_matches[rel_uri][uuid]:
                    uuid2sents.update({uuid: [correct_matches[rel_uri][uuid][lang]]})
                    cnt += 1
        sorted_uuids = sorted(list(uuid2sents.keys()))
        probing_result = probing_method(uuid2sents, probing_type=probing_type)
        with open(dump_fn, 'wb') as fp:
            pickle.dump((sorted_uuids, ranked_uuids, probing_result), fp)
        return sorted_uuids, ranked_uuids, probing_result

    def probe_uuids_by_lang(self, lang, cand_size=1000, by='probeless', probing_type='acts', reload=False, rel="placeholder"):
        """Cannot follow the design above as it requires large memory to save the active neuron values. So we have to use more efficient loader

        Args:
            cand_size (int, optional): The number of uuids we are going to use for neuron probing. Defaults to 100.
            by (str, optional): _description_. Defaults to 'probeless'.
            probing_type (str, optional): _description_. Defaults to 'acts'.
            reload (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert probing_type in ["acts", "outs"]
        lang2uuids, ranked_uuids = get_uuids_candidates_per_lang(self.dataset, cand_size)

        root_path = os.path.join(PROBER_ROOT_ON_DISK, self.dataset.model_name, 'probe_uuids')
        os.makedirs(root_path, exist_ok=True)
        sents_dump_fn = os.path.join(root_path, f'{cand_size}_lang2uuid2sents.pkl')
        if os.path.exists(sents_dump_fn) and reload==False:
            with open(sents_dump_fn, 'rb') as fp:
                lang2uuid2sents = pickle.load(fp)
        else:
            lang2uuid2sents = {}
            for rel_uri in tqdm(self.dataset.rels, desc=f"Generating sentences batch for probing neurons per uuid"):
                correct_matches = self.sent_iter.get_correct_matches(rel=rel_uri, match_type=self.match_type)
                for _lang in self.dataset.rel2langs[rel_uri]:
                    for obj_uri in correct_matches:
                        for uuid in correct_matches[obj_uri]:
                            if uuid in lang2uuids[_lang] and _lang in correct_matches[obj_uri][uuid]:
                                lang2uuid2sents.setdefault(_lang, {}).update({uuid: [correct_matches[obj_uri][uuid][_lang]]})
            with open(sents_dump_fn, 'wb') as fp:
                pickle.dump(lang2uuid2sents, fp)

        lang2uuids = {}
        uuid2matchedlangs = {}
        for _lang in tqdm(lang2uuid2sents.keys(), desc=f"Probing neurons across languages per uuid"):
            for uuid in lang2uuid2sents[_lang].keys():
                uuid2matchedlangs.setdefault(uuid, []).append(_lang)
            lang2uuids[_lang] = sorted(list(lang2uuid2sents[_lang].keys()))
        uuid2matchedlangs = {k: sorted(list(v)) for k, v in uuid2matchedlangs.items()}

        probing_method = self.get_probing_method(by)
        probing_dump_fn = os.path.join(root_path, f'{lang}_{cand_size}_lang2uuid2sents.pkl')
        if os.path.exists(probing_dump_fn) and reload==False:
            with open(probing_dump_fn, 'rb') as fp:
                probing_result = pickle.load(fp)
        else:
            probing_result = probing_method(lang2uuid2sents[lang], probing_type=probing_type)
            with open(probing_dump_fn, 'wb') as fp:
                pickle.dump(probing_result, fp)
        return uuid2matchedlangs, lang2uuids, ranked_uuids, probing_result

    def probeless(self, label2sents: dict, probing_type: str):
        """Run probeless method to probe activated neurons

        Args:
            label2sents (dict): labels to sentences, the neuron probing will be conducted based on average over all labels. 
                For example, if labels represent languages, the activated neurons for each language represent the neurons controlling the language code.
            probing_type (str): ['acts', 'outs']

        Returns:
            _type_: _description_
        """
        label_uris = sorted(list(label2sents.keys()))
        labels = []
        neurons = []
        label2idx = {}
        for label in label_uris:
            if len(label2sents[label])!=0:
                label2idx[label] = len(label2idx)
                for info in self.model.collect_neurons(label2sents[label], batchsize=self.batchsize, reload=False):
                    neurons.append(info[probing_type].mean(dim=1).flatten())
                    labels.append(label2idx[label])
        if not neurons:
            return None, None, None, None

        neurons = torch.stack(neurons).numpy()
        labels = np.asarray(labels)
        idx2label = {v:k for k, v in label2idx.items()}
        overall_ranking, ranking_per_tag, overall_summation, summation_per_tag = get_neuron_ordering_for_all_tags(neurons, labels, idx2label)
        return overall_ranking, ranking_per_tag, overall_summation, summation_per_tag

if __name__ == "__main__":

    dataset = MaskedDataset(model_name="mbert")
    sent_iter = SentIterator(dataset)
    prober = Prober(dataset, device='cuda:6', batchsize=128)
    # uuid2matchedlangs, lang2uuids, probing_result = prober.probe_uuids(rel="probe_uuids-place-holder", probing_type="acts")
    uuid2matchedlangs, lang2uuids, probing_result = prober.probe_uuids(rel="probe_uuids-place-holder", probing_type="outs")