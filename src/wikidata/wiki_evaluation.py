import sys
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from graelo_wiki import load_retraining_text_by_lang

sys.path.append('../../src/')
from mask_dataset import MaskedDataset
from pred_evaluation import get_all_and_matched_uuid_lsts
from utils import chunk_list, loader, chunk_list_by_value_range

def evaluate_corr_between_wikipages_and_p1(dataset: MaskedDataset):
    """ To show that the training data resource is not the only factor
    """
    from wiki_2018_dump import get_title_count_per_lang
    from pred_evaluation import calculate_overall_p1_score_standard
    lang2cnt = get_title_count_per_lang()
    lang2p1 = calculate_overall_p1_score_standard(dataset)

    sorted_lang = sorted(lang2cnt, key=lambda k: lang2cnt[k], reverse=False)
    # Extract values
    title_cnts = [lang2cnt[lang] for lang in sorted_lang]
    p1_scores = [round(lang2p1[lang], 4) for lang in sorted_lang]

    # Calculate correlation using numpy
    correlation = np.corrcoef(title_cnts, p1_scores)[0, 1]
    text_str = f'Correlation: {correlation:.2f}'


    # Your provided lists
    x = list(range(len(sorted_lang), 0, -1))
    langs = [dataset.display_lang(lang) for lang in sorted_lang]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Languages')
    ax1.set_ylabel('Wikipedia Page Count', color=color)
    ax1.plot(x, title_cnts, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(langs, rotation=90)

    # Create the second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Factual probing p1 score', color=color)
    ax2.plot(x, p1_scores, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Place text on figure. x=0.05 and y=0.95 places the text at the top left corner of the figure
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.title('Plot of Wikipedia page count and p1 score for 53 languages')
    plt.grid(True)
    plt.show()




""" Detect the knowledge that are transferred to low-resource languages. The idea is, finding the knowledge that
1. Both captured by language A and language B
2. The wikipedia page only shows in language A but not in language B
In this case, we will know this knowledge is transferred from A to B 
We hope to get the knowledge matrix, where the row represents the knowledge and column represents the languages. 

Two methods here:
1. get_wiki_matches_matrix_from_huggingface: use the wikipedia data from huggingface. It only has the latest data. THe methods is the title matching
2. get_wiki_matches_matrix_from_dumped_wiki_abstract: use 2018-11 dumped wiki data. It finds the co-occurence of subject-object as the singal for existence fo facutal knowledge.
3. get_wiki_matches_matrix_from_dumped_wiki_title: Use the 2018-11 dumped wiki data to check the existence of entities. This is done by matching the object/subject with wiki titles. 
        If an entity has corresponding titles, we assume that the factual knowledge exists.


Return: matrix per langauge
- Row: language
- Column: subject
- Value:
    - 0: without wiki data, not predicted
    - 1: with wiki data, not predicted
    - 2: without wiki data, predicted 
    - 3: with wiki data, predicted 
    - -1: not appeared in the prompt
"""
@loader
def get_wiki_matches_matrix_from_huggingface(dataset, reload=False):
    
    # Get the elements of all the rows - the subject uris
    all_sub_uris = set()
    lang2subs, lang2subs_uri = dataset.get_lang2subs()
    for lang in dataset.langs:
        all_sub_uris.update(lang2subs_uri[lang])
    all_sub_uris = sorted(list(all_sub_uris))
    langs = sorted(dataset.langs)

    # Initalize the matrix
    sub_matrix = np.ones((len(langs), len(all_sub_uris)), dtype = np.int8)
    sub_matrix = sub_matrix * -1
    
    for lang_idx, lang in tqdm(enumerate(langs), total=len(langs), desc="Generating wiki-probing matching matrix for analyzing cross-lingual transfer ability"):
        # Get all subjects that appear in language prompts
        all_subs = lang2subs[lang]
        all_subs_uri = lang2subs_uri[lang]

        # Get subjects that has wikipedia items
        
        sub2text = load_retraining_text_by_lang(dataset=dataset, lang=lang)[2]
        has_wiki_subs = list(sub2text.keys())
        
        # Get subjects that are correctly predicted during factual knowledge probing
        lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]
        uuid_info = dataset.get_uuid_info()
        predicted_subs = []
        for rel, uuids in lang2rel2matched_uuid[lang].items():
            predicted_subs.extend([uuid_info[rel][uuid]['sub'] for uuid in uuids])

        # Start to fill the value into the matrix
        for sub, sub_uri in zip(all_subs, all_subs_uri):
            sub_idx = all_sub_uris.index(sub_uri)
            if sub not in has_wiki_subs and sub not in predicted_subs:
                sub_matrix[lang_idx][sub_idx] = 0
            elif sub in has_wiki_subs and sub not in predicted_subs:
                sub_matrix[lang_idx][sub_idx] = 1
            elif sub not in has_wiki_subs and sub in predicted_subs:
                sub_matrix[lang_idx][sub_idx] = 2
            elif sub in has_wiki_subs and sub in predicted_subs:
                sub_matrix[lang_idx][sub_idx] = 3
    return all_sub_uris, langs, sub_matrix

@loader
def _get_title_object_subject_matchings(dataset: MaskedDataset, reload=False):
    from wiki_2018_dump import check_entity_existence_in_titles

    lang2obj, lang2objuri = dataset.get_lang2objs()
    lang2sub, lang2suburi = dataset.get_lang2subs()
    
    lang2sub2matches = {}
    lang2obj2matches = {}
    lang2labels = {}
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = []
        for lang in dataset.langs:
            futures.append(executor.submit(check_entity_existence_in_titles, lang2obj[lang], lang2sub[lang], lang))
        for ft in tqdm(as_completed(futures), total=len(futures), desc="Checking entity existence by looking at titles"):
            lang, obj_exist_label, sub_exist_label = ft.result()
            lang2labels[lang] = (obj_exist_label, sub_exist_label)
        
        for lang in dataset.langs:
            lang2sub2matches[lang] = {}
            for label_idx, label in enumerate(lang2labels[lang][1]):
                sub_uri = lang2suburi[lang][label_idx]
                lang2sub2matches[lang][sub_uri] = label
        for lang in dataset.langs:
            lang2obj2matches[lang] = {}
            for label_idx, label in enumerate(lang2labels[lang][0]):
                obj_uri = lang2objuri[lang][label_idx]
                lang2obj2matches[lang][obj_uri] = label
    return lang2obj2matches, lang2sub2matches

@loader
def get_wiki_matches_matrix_from_dumped_wiki_title(dataset: MaskedDataset, using_obj=False, reload=False):
    lang2obj2matches, lang2sub2matches = _get_title_object_subject_matchings(dataset)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]

    langs = sorted(dataset.langs)
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    sub_matrix = np.ones((len(langs), len(all_uuids)), dtype = np.int8)
    sub_matrix = sub_matrix * -1
    for uuid, uuid_info in tqdm(list(dataset.get_uuid_info_plain().items()), desc="Generating wiki-probing matching matrix for analyzing cross-lingual transfer ability"):
        for lang in uuid_info['langs']:
            lang_idx = langs.index(lang)
            uuid_idx = all_uuids.index(uuid)
            sub_uri = uuid_info['sub_uri']
            obj_uri = uuid_info['obj_uri']
            rel_uri = uuid_info['rel_uri']

            if using_obj:
                in_wiki = bool(min(lang2sub2matches[lang][sub_uri], lang2obj2matches[lang][obj_uri]))
            else:
                in_wiki = bool(lang2sub2matches[lang][sub_uri])

            matched = uuid in lang2rel2matched_uuid[lang][rel_uri]
            if not in_wiki and not matched:
                sub_matrix[lang_idx][uuid_idx] = 0
            elif in_wiki and not matched:
                sub_matrix[lang_idx][uuid_idx] = 1
            elif not in_wiki and matched:
                sub_matrix[lang_idx][uuid_idx] = 2
            elif in_wiki and matched:
                sub_matrix[lang_idx][uuid_idx] = 3
    
    return langs, all_uuids, sub_matrix
        
@loader
def _get_wiki_matches_resource_from_dumped_wiki_abstract(dataset, reload=False):
    """ To run the get_wiki_matches_matrix_by_dumped_wiki, we need several data structures
    1. Matching information of subject and object in wikidata. The information is saved as `suburi2lineid`, `objuri2lineid` per language
    2. The (subject-object) pair as factual knowledge. There are two types of factual knowledge:
        - all_sub_obj_pairs: all subjects and objects appeared in the MLAMA data
        - match_sub_obj_pairs: all subjects and objects that are predicted correctly by ML-LMs
    """
    import re

    from constants import WIKI_2018DUMP_CACHE
    root = os.path.join(WIKI_2018DUMP_CACHE, 'ggrep_result')
    lang2rel2all_uuid, lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)

    # Get the map from tokenized subject/object to their uri. 
    # E.g., {'en': {'Ja ##pan': Q150}}
    lang2tokenized2sub_uri = defaultdict(lambda: defaultdict(set))
    for sub_uri, lang2info in dataset.get_sub_info().items():
        for lang, info in lang2info.items():
            lang2tokenized2sub_uri[lang][' '.join(info['sub_tokens'])].add(sub_uri)
    lang2tokenized2obj_uri = defaultdict(lambda: defaultdict(set))
    for obj_uri, lang2info in dataset.get_obj_info().items():
        for lang, info in lang2info.items():
            lang2tokenized2obj_uri[lang][' '.join(info['obj_tokens'])].add(obj_uri)

    lang2matching_measurement = {}
    for lang in tqdm(dataset.langs, desc="Generating resource for measuring factual knowledge existence in wiki and matches by ML-LMs"):
        # Read ggrep result and analysis line-obj, line-sub information. 
        # Set the result in the dictionary with the format of {"Ja ##pan": [123, 456], ....}
        suburi2lineid = defaultdict(set)
        objuri2lineid = defaultdict(set)
        with open(os.path.join(root, lang, "subject_matches.txt"), 'r') as fp:
            for line in fp:
                line_id, *words = re.match(r'^(\d+):(.*)$', line.strip()).groups()
                for word in map(str.strip, words[0].split('\t')):
                    for sub_uri in lang2tokenized2sub_uri[lang][word]:
                        suburi2lineid[sub_uri].add(int(line_id))
        with open(os.path.join(root, lang, "object_matches.txt"), 'r') as fp:
            for line in fp:
                line_id, *words = re.match(r'^(\d+):(.*)$', line.strip()).groups()
                for word in map(str.strip, words[0].split('\t')):
                    for obj_uri in lang2tokenized2obj_uri[lang][word]:
                        objuri2lineid[obj_uri].add(int(line_id))

        # Read sub
        uuid_info = dataset.get_uuid_info()
        all_sub_obj_pairs = []
        match_sub_obj_pairs = []
        for rel, uuids in lang2rel2all_uuid[lang].items():
            all_sub_obj_pairs.extend([(uuid_info[rel][uuid]['sub_uri'], uuid_info[rel][uuid]['obj_uri']) for uuid in uuids])        
        for rel, uuids in lang2rel2matched_uuid[lang].items():
            match_sub_obj_pairs.extend([(uuid_info[rel][uuid]['sub_uri'], uuid_info[rel][uuid]['obj_uri']) for uuid in uuids])
        lang2matching_measurement[lang] = (suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs)
    return lang2matching_measurement

@loader
def _get_subject_object_cooccurence_in_abstract(dataset, reload=False):
    lang2matching_measurement = _get_wiki_matches_resource_from_dumped_wiki_abstract(dataset)
    lang2obj2sub2ids_cooc_allFK = {}
    lang2obj2sub2ids_cooc_matchedFK = {}
    for lang in tqdm(dataset.langs, desc="Retrieving object-subject pairs for all languages"):
        obj2sub2ids_allFK = {}
        obj2sub2ids_matchedFK = {}
        sub2id, obj2id, all_subobj, matched_subobj = lang2matching_measurement[lang] # sub2id: sub_uri2line_id, obj2id: obj_uri2line_id
        for sub, obj in all_subobj:
            if sub in sub2id and obj in obj2id:
                obj2sub2ids_allFK.setdefault(obj, {}).update({sub: sub2id[sub].intersection(obj2id[obj])})
        for sub, obj in matched_subobj:
            if sub in sub2id and obj in obj2id:
                obj2sub2ids_matchedFK.setdefault(obj, {}).update({sub: sub2id[sub].intersection(obj2id[obj])})
        lang2obj2sub2ids_cooc_allFK[lang] = obj2sub2ids_allFK
        lang2obj2sub2ids_cooc_matchedFK[lang] = obj2sub2ids_matchedFK
    return lang2obj2sub2ids_cooc_allFK, lang2obj2sub2ids_cooc_matchedFK

@loader
def get_wiki_matches_matrix_from_dumped_wiki_abstract(dataset, reload=False):
    lang2obj2sub2ids_cooc_allFK = _get_subject_object_cooccurence_in_abstract(dataset)[0]
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]

    # Initalize the matrix
    langs = sorted(dataset.langs)
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    sub_matrix = np.ones((len(langs), len(all_uuids)), dtype = np.int8)
    sub_matrix = sub_matrix * -1
    for uuid, uuid_info in tqdm(list(dataset.get_uuid_info_plain().items()), desc="Generating wiki-probing matching matrix for analyzing cross-lingual transfer ability"):
        for lang in uuid_info['langs']:
            lang_idx = langs.index(lang)
            uuid_idx = all_uuids.index(uuid)
            sub_uri = uuid_info['sub_uri']
            obj_uri = uuid_info['obj_uri']
            rel_uri = uuid_info['rel_uri']
            in_wiki = (
                obj_uri in lang2obj2sub2ids_cooc_allFK[lang]
                and sub_uri in lang2obj2sub2ids_cooc_allFK[lang][obj_uri]
                and lang2obj2sub2ids_cooc_allFK[lang][obj_uri][sub_uri]
            )
            matched = uuid in lang2rel2matched_uuid[lang][rel_uri]
            
            if not in_wiki and not matched:
                sub_matrix[lang_idx][uuid_idx] = 0
            elif in_wiki and not matched:
                sub_matrix[lang_idx][uuid_idx] = 1
            elif not in_wiki and matched:
                sub_matrix[lang_idx][uuid_idx] = 2
            elif in_wiki and matched:
                sub_matrix[lang_idx][uuid_idx] = 3
    
    return langs, all_uuids, sub_matrix

from collections import defaultdict
def defaultdict2set():
    return defaultdict(set)

def _get_subject_object_cooccurence_in_article(dataset, candidate_langs, reload=False):
    import pickle
    from wiki_2018_dump import TGT_DATA_ROOT, get_uri2file2lineids_from_article_grep_matching
    from wiki_2018_dump import _preprocess_uri2file2lineids_from_article_grep_matching
    ### Better to run _preprocess_uri2file2lineids_from_article_grep_matching in advance to get the line id information
    # _preprocess_uri2file2lineids_from_article_grep_matching(candidate_langs)

    cache_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'cache')
    os.makedirs(cache_root, exist_ok=True)
    all_uuid_info = dataset.get_uuid_info_per_lang()
    lang2uuids = {lang: all_uuid_info[lang] for lang in candidate_langs}
    
    lang2uuid2matches = {}
    for lang in candidate_langs:
        match_info_lang_fn = os.path.join(cache_root, f'{lang}_uuid2matches.pkl')
        if os.path.exists(match_info_lang_fn) and reload == False:
            with open(match_info_lang_fn, 'rb') as fp:
                lang2uuid2matches[lang] = pickle.load(fp)
            continue

        lang2uuid2matches[lang] = {}
        sub_grep_matching_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'sub', lang)
        obj_grep_matching_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'obj', lang)

        sub2file2ids = get_uri2file2lineids_from_article_grep_matching(sub_grep_matching_root, 'subject', reload=reload)
        obj2file2ids = get_uri2file2lineids_from_article_grep_matching(obj_grep_matching_root, 'object', reload=reload)

        uuids = lang2uuids[lang]
        for uuid in tqdm(uuids, desc=f"Finding matched subject-object in wiki article for language: {lang}"):
            sub_uri = uuid['sub_uri']
            obj_uri = uuid['obj_uri']
            all_files = set(sub2file2ids[sub_uri].keys()).intersection(set(obj2file2ids[obj_uri].keys()))
            lang2uuid2matches[lang][uuid['uuid']] = 0
            for fn in all_files:
                sub_lineids = sub2file2ids[sub_uri][fn]
                obj_lineids = obj2file2ids[obj_uri][fn]
                lang2uuid2matches[lang][uuid['uuid']] += len(sub_lineids.intersection(obj_lineids))
        with open(match_info_lang_fn, 'wb') as fp:
            pickle.dump(lang2uuid2matches[lang], fp)
    return lang2uuid2matches

def get_wiki_matches_matrix_from_dumped_wiki_article(dataset, candidate_langs=None, reload=False):
    candidate_langs = candidate_langs if candidate_langs else dataset.langs
    
    # TODO: Reload cooc takes time, need to manually run it before calling this function
    lang2uuid2wikicooc = _get_subject_object_cooccurence_in_article(dataset, candidate_langs, reload=False)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]

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

""" The code above can give the distribution of different types of knowledge. And the resource data to describe the co-occurrence. 
From here, there are functions that do analysis based on the data. 
1. plot_stacted_bar_and_percentage_of_fk_matching: plot the non-wiki & predicted and in-wiki predicted factual knowledge in a bar graph. 
2. analyze_niwp_information_by_lang: Used to capture non-wiki & predicted knowledge by abstract matching
3. display_niwp_fk_by_article_matching: Used to capture non-wiki & predicted knowledge by abstract matching
"""
def plot_stacted_bar_and_percentage_of_fk_matching(dataset, resource_type="article", candidate_langs=None):
    if resource_type not in ['abstract', 'title', 'article', 'tokenized_article']:
        raise ValueError(f"Unsupported resource_type: {resource_type}")
    
    if resource_type == 'abstract':
        langs, _, sub_matrix = get_wiki_matches_matrix_from_dumped_wiki_abstract(dataset)
    elif resource_type == 'title':
        langs, _, sub_matrix = get_wiki_matches_matrix_from_dumped_wiki_title(dataset)
    elif resource_type == 'article':
        langs, _, sub_matrix = get_wiki_matches_matrix_from_dumped_wiki_article(dataset, candidate_langs)
    elif resource_type == 'tokenized_article':
        langs, _, sub_matrix = get_wiki_matches_matrix_from_dumped_wiki_article(dataset, candidate_langs)
    else:
        raise ValueError(f"Unsupported resource_type: {resource_type}")

    matched_cnt = np.array([len(np.where(sub_matrix[row_id]>=2)[0]) for row_id in range(len(langs))])
    sorted_langs = [langs[idx] for idx in np.argsort(matched_cnt)][::-1]
    matrix_info = {}
    for lang in sorted_langs:
        row_id = langs.index(lang)
        matrix_info[dataset.display_lang(lang)] = {
            "Not in evaluation dataset": len(np.where(sub_matrix[row_id]==-1)[0]),
            "Has no wiki & Not predicted": len(np.where(sub_matrix[row_id]==0)[0]),
            "Has wiki & Not predicted": len(np.where(sub_matrix[row_id]==1)[0]),
            "Has no wiki & Predicted": len(np.where(sub_matrix[row_id]==2)[0]),
            "Has wiki & Predicted": len(np.where(sub_matrix[row_id]==3)[0])
        }

    df = pd.DataFrame(matrix_info).T
    df = df.reset_index().rename(columns={'index': 'language'})
    df = df.drop(["Not in evaluation dataset"], axis=1)
    df = df.drop(["Has no wiki & Not predicted"], axis=1)
    df = df.drop(["Has wiki & Not predicted"], axis=1)
    df['Rate of no-wiki data'] = df['Has no wiki & Predicted'] / (df['Has no wiki & Predicted'] + df['Has wiki & Predicted'])

    fig, ax1 = plt.subplots(figsize=(13, 8))
    ax1.bar(df['language'], df['Has no wiki & Predicted'], label='Has no wiki & Predicted')
    ax1.bar(df['language'], df['Has wiki & Predicted'], bottom=df['Has no wiki & Predicted'], label='Has wiki & Predicted')
    ax1.legend()
    ax1.tick_params(axis='y')
    ax1.set_xticklabels(df['language'], rotation=60)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Not-in-wiki predicted FK')
    ax2.plot(df['language'], df['Rate of no-wiki data'], marker='o', linestyle='-')
    ax2.tick_params(axis='y')

    ax1.set_title('Distribution of 2 types of factual knowledge (subject) by two dimension - (in or not in wiki) & (correctly predicted or not)')
    fig.tight_layout()
    plt.show()

def analyze_niwp_information_by_lang(lang, dataset, all_uuids, langs, sub_matrix):
    """ niwp: the factual knowledge not occurred in wiki data but is corrected predicted
    Usage:
        ```
            langs, all_uuids, sub_matrix = get_wiki_matches_matrix_from_dumped_wiki_abstract(dataset)
            rel_niwp, obj_niwp, sub_niwp, rel_niwp_counter, obj_niwp_counter, sub_niwp_counter = /
                get_niwp_information_by_lang(lang, dataset, all_uuids, langs, sub_matrix)
            
            lang2matching_measurement = _get_wiki_matches_resource_from_dumped_wiki_abstract(dataset)
            suburi2lineid, objuri2lineid, all_sub_obj_pairs, match_sub_obj_pairs = lang2matching_measurement['ja']

            print(list(sorted_obj_counter.keys()))

            uuid_info = dataset.get_uuid_info_all_lang()
            for uuid in list(obj_niwp['dragon'])[:10]:               
                print(f"sub: {uuid_info[uuid]['ja']['sub']} \nrel: {uuid_info[uuid]['ja']['rel']} \nobj: {uuid_info[uuid]['ja']['obj']} \nOccurence of subject: {len(suburi2lineid[uuid_info[uuid]['ja']['sub_uri']])}\n")
        ```
        >>> sub: ヨーロッパの竜 
            rel: [X]は[Y]のサブクラスです。 
            obj: 竜 
            Occurence of subject: 0
    """
    
    """Get the number of FKs that has no wiki across all languages but get correctly predicted somewhere. """
    pred = np.where(sub_matrix>=2)[1]
    with_wiki_pred = np.where(sub_matrix==3)[1]
    print(f"The number of factual knowledge that has no wiki data across all knowledge but get correctly predicted is: {len(set(list(pred))) - len(set(list(with_wiki_pred)))}")
    
    uuid_info = dataset.get_uuid_info_plain()
    en_row = sub_matrix[langs.index(lang)]
    niw_pred = np.where(en_row==2) # Not in wikidata but correctly predicted
    niwp_uuids = [all_uuids[idx] for idx in niw_pred[0]]

    rel_niwp = defaultdict(set)
    obj_niwp = defaultdict(set)
    sub_niwp = defaultdict(set)
    rel_niwp_counter = defaultdict(int)
    obj_niwp_counter = defaultdict(int)
    sub_niwp_counter = defaultdict(int)

    for uuid in niwp_uuids:
        rel_niwp[uuid_info[uuid]['rel']].add(uuid)
        obj_niwp[uuid_info[uuid]['obj']].add(uuid)
        sub_niwp[uuid_info[uuid]['sub']].add(uuid)
        rel_niwp_counter[uuid_info[uuid]['rel']] += 1
        obj_niwp_counter[uuid_info[uuid]['obj']] += 1
        sub_niwp_counter[uuid_info[uuid]['sub']] += 1
        
    rel_niwp_counter = dict(sorted(rel_niwp_counter.items(), key=lambda item: item[1], reverse=True))
    obj_niwp_counter = dict(sorted(obj_niwp_counter.items(), key=lambda item: item[1], reverse=True))
    sub_niwp_counter = dict(sorted(sub_niwp_counter.items(), key=lambda item: item[1], reverse=True))
    return rel_niwp, obj_niwp, sub_niwp, rel_niwp_counter, obj_niwp_counter, sub_niwp_counter

def display_niwp_fk_by_article_matching(dataset, lang):
    lang2uuid2wiki_matches = _get_subject_object_cooccurence_in_article(dataset, dataset.langs)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]
    uuid_info = dataset.get_uuid_info_all_lang()
    all_cnt = 0
    overlap_cnt = 0
    for uuid in uuid_info:
        if lang not in uuid_info[uuid]:
            continue
        sub = uuid_info[uuid][lang]['sub']
        obj = uuid_info[uuid][lang]['obj']
        rel = uuid_info[uuid][lang]['rel']
        predicted = any([uuid in lang2rel2matched_uuid[lang][rel] for rel in lang2rel2matched_uuid[lang]])
        if predicted:
            if uuid not in lang2uuid2wiki_matches[lang] or lang2uuid2wiki_matches[lang][uuid] == 0:
                print(f"sub: {sub}, obj: {obj}, rel: {rel}")

@loader
def _get_correct_wrong_prediction_of_inwiki_fk(dataset, reload=False):
    lang2uuid2matches = _get_subject_object_cooccurence_in_article(dataset, dataset.langs)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]
    lang2matched_uuid = {}
    for lang in lang2rel2matched_uuid:
        lang2matched_uuid[lang] = []
        for rel in lang2rel2matched_uuid[lang]:
            lang2matched_uuid[lang].extend(lang2rel2matched_uuid[lang][rel])    

    cnt2corrt_cnt = defaultdict(int)
    cnt2wrong_cnt = defaultdict(int)

    all_cnts = set()
    for lang in tqdm(lang2uuid2matches):
        for uuid in lang2uuid2matches[lang]:
            match_cnt = lang2uuid2matches[lang][uuid]
            all_cnts.add(match_cnt)
            if uuid in lang2matched_uuid[lang]:
                cnt2corrt_cnt[match_cnt] += 1
            else:
                cnt2wrong_cnt[match_cnt] += 1
    all_cnts = sorted(list(all_cnts))
    corrt_cnts = [cnt2corrt_cnt[cnt] for cnt in all_cnts]
    wrong_cnts = [cnt2wrong_cnt[cnt] for cnt in all_cnts]
    return all_cnts, corrt_cnts, wrong_cnts

def plot_stacked_bar_and_percentage_of_inwiki_prediction(dataset, chunk_size=50, max_cnt=2500):
    raw_all_cnts, raw_corrt_cnts, raw_wrong_cnts = _get_correct_wrong_prediction_of_inwiki_fk(dataset)
    _, chunked_range, chunked_idx = chunk_list_by_value_range(raw_all_cnts[1:], chunk_size=chunk_size, max_val=max_cnt)
    x_labels = [f"{sidx + 1} ~ {eidx + 1}" for sidx, eidx in chunked_range]

    corrt_cnts = []
    wrong_cnts = []
    for idxs in chunked_idx:
        corrt_cnts.append(sum(raw_corrt_cnts[idx] for idx in idxs))
        wrong_cnts.append(sum(raw_wrong_cnts[idx] for idx in idxs))

    corrt_rate = [corrt_cnts[i]/(corrt_cnts[i] + wrong_cnts[i]) for i in range(len(corrt_cnts))]

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.bar(x_labels, corrt_cnts, label='in-wiki & predicted')
    ax1.bar(x_labels, wrong_cnts, bottom=corrt_cnts, label='in-wiki & non-predicted')
    ax1.legend()
    ax1.tick_params(axis='y')
    ax1.set_xticklabels(x_labels, rotation=90)
    
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('The percentage of inwiki & predicted FK')
    ax2.plot(x_labels, corrt_rate, marker='o', linestyle='-')
    ax2.tick_params(axis='y')
    ax1.set_title('Distribution of wiki-occurred factual knowledge: predicted vs non-predicted')
    ax2.set_xlabel('The number of co-occurrences for subject-object')
    fig.tight_layout()
    plt.show()

def measure_crosslingual_transfer_by_correlation_corrtpred_inwiki_fk(dataset, per_lang=True, reload=False):
    """ The idea: If Knowledge A non-wiki & predicted, but the knowledge is described in language B. It's possible that the knowledge is transferred from B to A.
    """
    from pred_evaluation import calculate_overall_p1_score_standard

    lang2uuid2matches = _get_subject_object_cooccurence_in_article(dataset, dataset.langs)
    lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset=dataset, reload=False)[1]

    lang2nonwiki_pred_uuid = {}
    lang2inwiki_pred_uuid = {}
    for lang in lang2rel2matched_uuid:
        lang2nonwiki_pred_uuid[lang] = set()
        lang2inwiki_pred_uuid[lang] = set()
        for rel in lang2rel2matched_uuid[lang]:
            for uuid in lang2rel2matched_uuid[lang][rel]:
                if lang2uuid2matches[lang][uuid] > 0:
                    lang2inwiki_pred_uuid[lang].add(uuid)
                else:
                    lang2nonwiki_pred_uuid[lang].add(uuid)
    sorted_langs, _ = calculate_overall_p1_score_standard(dataset)

    lang2rate = {}
    if per_lang:
        for test_lang in sorted_langs:
            lang2rate[test_lang] = []
            for base_lang in sorted_langs:
                nonwiki_pred_in_test_lang = lang2nonwiki_pred_uuid[test_lang]
                inwiki_pred_in_base_lang = lang2inwiki_pred_uuid[base_lang]
                fk_nonwiki_test_inwik_base_cnt = len(inwiki_pred_in_base_lang.intersection(nonwiki_pred_in_test_lang))
                fk_nonwiki_test_cnt = len(nonwiki_pred_in_test_lang)
                lang2rate[test_lang].append(fk_nonwiki_test_inwik_base_cnt/fk_nonwiki_test_cnt)
    else:
        for test_lang in sorted_langs:
            nonwiki_pred_in_test_lang = lang2nonwiki_pred_uuid[test_lang]
            lang2rate[test_lang] = 0
            for uuid in nonwiki_pred_in_test_lang:
                for base_lang in sorted_langs:
                    if uuid in lang2inwiki_pred_uuid[base_lang]:
                        lang2rate[test_lang] += 1
                        break
            lang2rate[test_lang] /= len(nonwiki_pred_in_test_lang)
    return sorted_langs, lang2rate





if __name__ == "__main__":
    dataset = MaskedDataset(model_name="mbert")
    # _get_subject_object_cooccurence_in_article(
    #     dataset=dataset,
    #     candidate_langs=dataset.langs,
    #     reload=True)
    _get_subject_object_cooccurence_in_tokenized_article(dataset)
    