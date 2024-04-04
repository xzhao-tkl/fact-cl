import os
import pickle
import re
import sys
import numpy as np
sys.path.append('../../src/')

from utils import loader

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pred_evaluation import get_all_and_matched_uuid_lsts

from mask_dataset import MaskedDataset
from wiki_2018_dump import TGT_DATA_ROOT

    

def return_allthings(lang, tokenized2sub_uri, tokenized2obj_uri, rel2all_uuid, rel2matched_uuid):
    root = "/home/xzhao/workspace/probing-mulitlingual/datasets/2018_dump_wiki_cache/sub_obj_ggrep"
    dump_root = "/disk/xzhao/datasets/wikipedia_2018_octnov/sub_obj_ggrep_processed_result_dump"
    os.makedirs(dump_root, exist_ok=True)

    text2id = {}
    match_dump_fn = os.path.join(dump_root, f"{lang}_matches.pkl")
    id_dump_fn = os.path.join(dump_root, f"{lang}_text2ids.pkl")
    if os.path.exists(match_dump_fn) and os.path.exists(id_dump_fn):
        with open(id_dump_fn, 'rb') as fp:
            text2id = pickle.load(fp)
        with open(match_dump_fn, 'rb') as fp:
            suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs = pickle.load(fp)
        return lang, text2id, suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs

    lang_root = os.path.join(root, lang)
    suburi2lineid = defaultdict(set)
    objuri2lineid = defaultdict(set)

    for subdir in os.listdir(lang_root):
        subdir = os.path.join(lang_root, subdir)
        for idx in range(10):
            obj_match_fn = os.path.join(subdir, f"object_match.txt.0{idx}")
            sub_match_fn = os.path.join(subdir, f"subject_match.txt.0{idx}")            
            file_id = subdir.split("/")[-1] + f":{idx}"
            with open(sub_match_fn, 'r') as fp:
                for line in fp:
                    line_id, *words = re.match(r'^(\d+):(.*)$', line.strip()).groups()
                    for word in map(str.strip, words[0].split('\t')):
                        for sub_uri in tokenized2sub_uri[word]:
                            unique_text = f"{file_id}:{line_id}"
                            if unique_text not in text2id:
                                text2id.update({unique_text: len(text2id)})
                            suburi2lineid[sub_uri].add(text2id[unique_text])
            with open(obj_match_fn, 'r') as fp:
                for line in fp:
                    line_id, *words = re.match(r'^(\d+):(.*)$', line.strip()).groups()
                    for word in map(str.strip, words[0].split('\t')):
                        for obj_uri in tokenized2obj_uri[word]:
                            unique_text = f"{file_id}:{line_id}"
                            if unique_text not in text2id:
                                text2id.update({unique_text: len(text2id)})
                            objuri2lineid[obj_uri].add(text2id[unique_text])

    # Read sub
    uuid_info = dataset.get_uuid_info()
    all_sub_obj_pairs = []
    match_sub_obj_pairs = []
    for rel, uuids in rel2all_uuid.items():
        all_sub_obj_pairs.extend([(uuid_info[rel][uuid]['sub_uri'], uuid_info[rel][uuid]['obj_uri']) for uuid in uuids])        
    for rel, uuids in rel2matched_uuid.items():
        match_sub_obj_pairs.extend([(uuid_info[rel][uuid]['sub_uri'], uuid_info[rel][uuid]['obj_uri']) for uuid in uuids])
    with open(match_dump_fn, 'wb') as fp:
        pickle.dump((suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs), fp)
    with open(id_dump_fn, 'wb') as fp:
        pickle.dump(text2id, fp)
    return lang, text2id, suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs

def get_wiki_matches_resource_from_tokenized_wiki_article(dataset):
    lang2rel2all_uuid, lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)
    lang2tokenized2sub_uri = defaultdict(lambda: defaultdict(set))
    for sub_uri, lang2info in dataset.get_sub_info().items():
        for lang, info in lang2info.items():
            lang2tokenized2sub_uri[lang][' '.join(info['sub_tokens'])].add(sub_uri)
    lang2tokenized2obj_uri = defaultdict(lambda: defaultdict(set))
    for obj_uri, lang2info in dataset.get_obj_info().items():
        for lang, info in lang2info.items():
            lang2tokenized2obj_uri[lang][' '.join(info['obj_tokens'])].add(obj_uri)

    with ProcessPoolExecutor(max_workers=53) as executor:
        futures = []
        for lang in tqdm(dataset.langs, desc="Generating resource for measuring factual knowledge existence in wiki and matches by ML-LMs"):
            futures.append(executor.submit(return_allthings, lang, lang2tokenized2sub_uri[lang], lang2tokenized2obj_uri[lang], lang2rel2all_uuid[lang], lang2rel2matched_uuid[lang]))
        for ft in tqdm(as_completed(futures)):
            lang, text2id, suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs = ft.result()
            yield lang, text2id, suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs

@loader
def get_subject_object_cooccurence_in_tokenized_article(dataset, reload=False):
    cache_root = os.path.join(TGT_DATA_ROOT, "sub_obj_ggrep_processed_result_dump", 'cache')
    os.makedirs(cache_root, exist_ok=True)
    all_uuid_info = dataset.get_uuid_info_per_lang()
    lang2uuids = {lang: all_uuid_info[lang] for lang in dataset.langs}
    
    lang2uuid2matches = {}
    lang2matching_measurement_iterator = get_wiki_matches_resource_from_tokenized_wiki_article(dataset)
    for res in tqdm(lang2matching_measurement_iterator, desc="Retrieving object-subject pairs for all languages"):
        lang, _, sub2id, obj2id, _, _ = res
        match_info_lang_fn = os.path.join(cache_root, f'{lang}_uuid2matches.pkl')
        if os.path.exists(match_info_lang_fn) and reload == False:
            with open(match_info_lang_fn, 'rb') as fp:
                lang2uuid2matches[lang] = pickle.load(fp)
            continue

        lang2uuid2matches[lang] = {}
        uuids = lang2uuids[lang]
        for uuid in tqdm(uuids, desc=f"Finding matched subject-object in tokenized wiki article for language: {lang}"):
            sub_uri = uuid['sub_uri']
            obj_uri = uuid['obj_uri']
            shared_ids = set(sub2id[sub_uri]).intersection(set(obj2id[obj_uri]))
            lang2uuid2matches[lang][uuid['uuid']] = len(shared_ids)
        with open(match_info_lang_fn, 'wb') as fp:
            pickle.dump(lang2uuid2matches[lang], fp)
    return lang2uuid2matches

@loader
def get_wiki_matches_matrix_from_tokenized_wiki_article(dataset, candidate_langs=None, reload=False):
    candidate_langs = candidate_langs if candidate_langs else dataset.langs
    
    # TODO: Reload cooc takes time, need to manually run it before calling this function
    lang2uuid2wikicooc = get_subject_object_cooccurence_in_tokenized_article(dataset, candidate_langs, reload=False)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=True)[1]

    # Initalize the matrix
    langs = sorted(candidate_langs)
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    sub_matrix = np.ones((len(langs), len(all_uuids)), dtype = np.int8)
    sub_matrix = sub_matrix * -1
    for uuid, uuid_info in tqdm(list(dataset.get_uuid_info_plain().items()), desc="Generating wiki-probing matching matrix for analyzing cross-lingual transfer ability"):
        for lang in uuid_info['langs']:
            if lang not in candidate_langs:
                continue
            lang_idx = langs.index(lang)
            uuid_idx = all_uuids.index(uuid)
            rel_uri = uuid_info['rel_uri']

            if uuid in lang2uuid2wikicooc[lang] and lang2uuid2wikicooc[lang][uuid] > 0:
                in_wiki = True    
            else:
                in_wiki = False
            predicted = uuid in lang2rel2matched_uuid[lang][rel_uri]
            
            if not in_wiki and not predicted:
                sub_matrix[lang_idx][uuid_idx] = 0
            elif in_wiki and not predicted:
                sub_matrix[lang_idx][uuid_idx] = 1
            elif not in_wiki and predicted:
                sub_matrix[lang_idx][uuid_idx] = 2
            elif in_wiki and predicted:
                sub_matrix[lang_idx][uuid_idx] = 3
    
    return langs, all_uuids, sub_matrix

if __name__ == "__main__":
    dataset = MaskedDataset(model_name="mbert", reload=False)
    get_wiki_matches_resource_from_tokenized_wiki_article(dataset)
    get_subject_object_cooccurence_in_tokenized_article(dataset, reload=True)
    get_wiki_matches_matrix_from_tokenized_wiki_article(dataset, reload=True)