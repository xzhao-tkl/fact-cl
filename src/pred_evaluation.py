import ast
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from langcodes import Language
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from constants import EVALUATION_ROOT, PREDICTION_ROOT
from mask_dataset import MaskedDataset
from utils import (language_distance, language_distance_matrix, loader,
                   strip_space)


def read_pred(lang, rel, root):
    tgt_csv_fn = os.path.join(root, lang, f"{lang}-{rel}.csv")
    if not os.path.exists(tgt_csv_fn):
        return pd.DataFrame()
    return pd.read_csv(tgt_csv_fn, index_col=0)

"""Evaluate p1 score by two methods: full match and partial match.
"""
def full_match(gold, evl):
    match_cnt = 0
    matches = []
    for idx in gold.index:
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]["obj_ids"]), is_wrapped=False)
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = [
            strip_space(ast.literal_eval(ids_str))
            for ids_str in evl[evl["id"] == idx]["pred_ids"].tolist()
        ]
        if obj_ids in preds:
            matches.append(idx)
            match_cnt += 1
    return match_cnt, matches


def partial_match(gold, evl):
    match_cnt = 0
    matches = []
    for idx in gold.index:
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]["obj_ids"]))
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = [
            strip_space(ast.literal_eval(ids_str))
            for ids_str in evl[evl["id"] == idx]["pred_ids"].tolist()
        ]
        for pred_ids in preds:
            if all(x in pred_ids for x in obj_ids):
                matches.append(idx)
                match_cnt += 1
                break
    return match_cnt, matches


def evaluate_p1_by_language(dataset, lang) -> dict:
    query_cnt = 0
    full_match_cnt = 0
    partial_match_cnt = 0
    p1_per_rel_full = {}
    p1_per_rel_partial = {}
    queries_per_rel = {}
    for rel in dataset.get_rels_in_lang(lang):
        full_match_cnt_by_rel = 0
        partial_match_cnt_by_rel = 0
        try:
            gold = dataset.get_lang_type(lang, rel)
            evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
            full_match_cnt_by_rel = full_match(gold, evl)[0]
            # partial_match_cnt_by_rel = partial_match(gold, evl)[0]
            full_match_cnt += full_match_cnt_by_rel
            # partial_match_cnt += partial_match_cnt_by_rel
            query_cnt += len(gold)
            if len(gold) != 0:
                p1_per_rel_full[rel] = round(full_match_cnt_by_rel/len(gold), 6)
                # p1_per_rel_partial[rel] = round(partial_match_cnt_by_rel/len(gold), 6)
            else:
                p1_per_rel_full[rel] = 0
                # p1_per_rel_partial[rel] = 0
            queries_per_rel[rel] = len(gold)
        except Exception as e:
            print(f"Error occurs for {lang}-{rel}, {e}")
    full_p1 = round(full_match_cnt / query_cnt, 6)
    # partial_p1 = round(partial_match_cnt / query_cnt, 6)
    return {
        lang: {
                "full-match": full_p1, 
                # "partial-match": partial_p1, 
                "full-match-per-rel": p1_per_rel_full, 
                # "partial-match-per-rel": p1_per_rel_partial,
                "queries_per_rel": queries_per_rel}}

@loader
def p1_evaluate_parallel(dataset, thread_num=56, reload=False, verbose=True):
    """Evaluate p1 matching score for all relations in each language and return the gathered inforamtion
    The per-relation p1 scores for each language is draw in a figure.

    Returns:
        p1_info: {
            lang: {
                "full-match": "sclar", 
                "partial-match": "sclar", 
                "full-match-per-rel": {rel: sclar}, 
                "partial-match-per-rel": {rel: sclar},
                "queries_per_rel": {rel: sclar}
            }
        }
    """
    # sourcery skip: dict-assign-update-to-union
    p1_info = {}
    with ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [
            executor.submit(evaluate_p1_by_language, dataset, lang)
            for lang in dataset.langs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running p1_evaluate_parallel for model {dataset.model_name}"):
            p1_info.update(future.result())
    # draw_p1_distri_per_lang(p1_info)
    return p1_info

"""Load and save the matrix of probing result. 
- Matching Matrix: object_number x language_number
    Each cell represents the number of factual knowledge that are correctly predicted for specific object. 
- Probing Matrix: object_number x language_number
    Each cell represents the number of factual knowledge the probing dataset has.
"""

def get_full_match_cnt_per_obj(gold, evl):
    obj2match_cnt = {}
    for idx, row in gold.iterrows():
        obj_ids = row.obj_ids if isinstance(row.obj_ids, list) else ast.literal_eval(row.obj_ids)
        obj_ids = strip_space(obj_ids, is_wrapped=False)
        preds = []
        for ids in evl[evl["uuid"] == row.uuid]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append(strip_space(ids, is_wrapped=False))
        
        if obj_ids in preds:
            if row.obj_uri in obj2match_cnt:
                obj2match_cnt[row.obj_uri] += 1
            else:
                obj2match_cnt[row.obj_uri] = 1
    return obj2match_cnt

def get_full_match_per_uuid(gold, evl):
    uuid2match = {}
    for idx, row in gold.iterrows():
        obj_ids = row.obj_ids if isinstance(row.obj_ids, list) else ast.literal_eval(row.obj_ids)
        obj_ids = strip_space(obj_ids, is_wrapped=False)
        preds = []
        for ids in evl[evl["uuid"] == row.uuid]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append(strip_space(ids))
        if obj_ids in preds:
            uuid2match[row.uuid] = True
        else:
            uuid2match[row.uuid] = False
    return uuid2match

def get_exact_match_per_uuid(gold, evl):
    uuid2match = {}
    for idx, row in gold.iterrows():
        obj_ids = row.obj_ids if isinstance(row.obj_ids, list) else ast.literal_eval(row.obj_ids)
        preds = []
        for ids in evl[evl["uuid"] == row.uuid]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append([_id[0] for _id in ids])
        if obj_ids in preds:
            uuid2match[row.uuid] = True
        else:
            uuid2match[row.uuid] = False
    return uuid2match

def get_partial_match_per_uuid(gold, evl):
    uuid2match = {}
    for idx, row in gold.iterrows():
        obj_ids = row.obj_ids if isinstance(row.obj_ids, list) else ast.literal_eval(row.obj_ids)
        obj_ids = strip_space(obj_ids, is_wrapped=False)
        preds = []
        for ids in evl[evl["uuid"] == row.uuid]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append(strip_space(ids))
        for pred_ids in preds:
            if all(x in pred_ids for x in obj_ids):
                uuid2match[row.uuid] = True
                break
        if row.uuid not in uuid2match:
            uuid2match[row.uuid] = False
    return uuid2match

@loader
def get_gold_matrix_per_obj(dataset: MaskedDataset, reload=False) -> tuple[list[str], list[str], np.ndarray]:
    rel_obj_pairs = dataset.get_rel_obj_pairs()
    langs = sorted(list(dataset.langs))
    matrix = np.zeros((len(langs), len(rel_obj_pairs)))
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        for idx, row in gold.iterrows():    
            obj_idx = rel_obj_pairs.index(f"{rel}-{row.obj_uri}")
            matrix[lang_idx][obj_idx] += 1
    return langs, rel_obj_pairs, matrix

@loader
def get_gold_matrix_per_uuid(dataset: MaskedDataset, reload=False) -> tuple[list[str], list[str], np.ndarray]:
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    langs = sorted(list(dataset.langs))
    matrix = np.zeros((len(langs), len(all_uuids)))
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        for idx, row in gold.iterrows():    
            uuid = all_uuids.index(f"{row.uuid}")
            matrix[lang_idx][uuid] = 1
    return langs, all_uuids, matrix

@loader
def get_full_match_matrix_by_obj(dataset: MaskedDataset, reload=False):
    rel_obj_pairs = dataset.get_rel_obj_pairs()
    langs = sorted(dataset.langs)
    matrix = np.zeros((len(langs), len(rel_obj_pairs)))
        
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
        try:
            obj2match_cnt = get_full_match_cnt_per_obj(gold, evl)
            for obj_uri in obj2match_cnt.keys():
                obj_idx = rel_obj_pairs.index(f"{rel}-{obj_uri}")
                matrix[lang_idx][obj_idx] = obj2match_cnt[obj_uri]
        except Exception as e:
            print(lang, rel)
            raise e
    return langs, rel_obj_pairs, matrix

@loader
def get_partial_match_matrix_by_uuid(dataset: MaskedDataset, reload=False):
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    langs = sorted(dataset.langs)
    matrix = np.zeros((len(langs), len(all_uuids)))
        
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
        uuid2match = get_partial_match_per_uuid(gold, evl)
        for uuid in uuid2match.keys():
            uuid_idx = all_uuids.index(uuid)
            matrix[lang_idx][uuid_idx] = int(uuid2match[uuid])
    return langs, all_uuids, matrix

@loader
def get_full_match_matrix_by_uuid(dataset: MaskedDataset, reload=False):
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    langs = sorted(dataset.langs)
    matrix = np.zeros((len(langs), len(all_uuids)))
        
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
        try:
            uuid2match = get_full_match_per_uuid(gold, evl)
        except Exception as e:
            print(rel, lang)
            raise e
        for uuid in uuid2match.keys():
            uuid_idx = all_uuids.index(uuid)
            matrix[lang_idx][uuid_idx] = int(uuid2match[uuid])
    return langs, all_uuids, matrix

@loader
def get_exact_match_matrix_by_uuid(dataset: MaskedDataset, reload=False):
    all_uuids = sorted(list(dataset.get_uuid_info_plain().keys()))
    langs = sorted(dataset.langs)
    matrix = np.zeros((len(langs), len(all_uuids)))
        
    for lang, rel in tqdm(list(dataset.lang_rel_iter())):
        lang_idx = langs.index(lang)
        gold = dataset.get_lang_type(lang, rel)
        evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
        try:
            uuid2match = get_exact_match_per_uuid(gold, evl)
        except Exception as e:
            print(rel, lang)
            raise e
        for uuid in uuid2match.keys():
            uuid_idx = all_uuids.index(uuid)
            matrix[lang_idx][uuid_idx] = int(uuid2match[uuid])
    return langs, all_uuids, matrix

def calculate_macro_p1_score_from_match_matrix(pred_matrix, langs, dataset):
    _langs, all_uuids, gold_matrix = get_gold_matrix_per_uuid(dataset)
    assert _langs == langs
    
    rel2idx = {}
    for rel in dataset.uuid_info:
        rel2idx[rel] = [all_uuids.index(uuid) for uuid in dataset.uuid_info[rel].keys()]

    rel2p1 = {}
    lang2macro_p1 = {lang: [] for lang in langs}
    for rel in rel2idx:
        full_match_by_rel = pred_matrix[:, rel2idx[rel]].sum(axis=1)
        gold_match_by_rel = gold_matrix[:, rel2idx[rel]].sum(axis=1)
        rel2p1[rel] = np.divide(full_match_by_rel, gold_match_by_rel)
        for idx, lang in enumerate(langs):
            if not np.isnan(rel2p1[rel][idx]):
                lang2macro_p1[lang].append(rel2p1[rel][idx])
    lang2macro_p1 = {lang:sum(p1s)/len(p1s) for lang, p1s in lang2macro_p1.items()}
    return lang2macro_p1

def calculate_overall_p1_score_from_match_matrix(pred_matrix, langs, dataset):
    _langs, _, gold_matrix = get_gold_matrix_per_uuid(dataset)
    assert _langs == langs
    p1_scores = np.nansum(pred_matrix, axis=1)/np.nansum(gold_matrix, axis=1)
    return {langs[idx]: p1 for idx, p1 in enumerate(p1_scores)}
    
def calculate_relwise_p1_score_from_match_matrix(dataset, match_matrix=None, target_langs=None):
    _langs, _, gold_matrix = get_gold_matrix_per_uuid(dataset)
    assert _langs == langs
    
    rel2idx = {}
    for idx, rel_obj in enumerate(relobjs):
        rel2idx.setdefault(rel_obj.split('-')[0], []).append(idx)
    
    rel2p1 = {}
    for rel in rel2idx:
        _gold = np.nansum(gold_matrix[:, rel2idx[rel]], axis=1)
        _match = np.nansum(match_matrix[:, rel2idx[rel]], axis=1) 
        rel2p1[rel] = np.where(_gold==0, np.inf, _match/_gold)
    return target_langs, rel2p1

def calculate_objwise_score_from_match_matrix(pred_matrix, langs, dataset: MaskedDataset, ignore_rel=False):
    _langs, all_uuids, gold_matrix = get_gold_matrix_per_uuid(dataset)
    assert _langs == langs

    obj2idx = {obj_uri:[] for obj_uri in dataset.obj_info}

    for idx, uuid in enumerate(all_uuids):
        obj_uri = dataset.uuid_info_plain[uuid]['obj_uri']
        obj2idx[obj_uri].append(idx)

    lang2obj2p1 = {lang: {} for lang in langs}
    for obj in obj2idx:
        full_match_by_obj = pred_matrix[:, obj2idx[obj]].sum(axis=1)
        gold_match_by_obj = gold_matrix[:, obj2idx[obj]].sum(axis=1)
        for lang_idx, lang in enumerate(langs):
            if gold_match_by_obj[lang_idx] > 0:
                lang2obj2p1[lang][obj] = (full_match_by_obj[lang_idx], gold_match_by_obj[lang_idx])
    return lang2obj2p1

def calculate_overall_p1_score_standard(dataset):
    test_langs, _, match_matrix = get_full_match_matrix_by_uuid(dataset, reload=False)
    return calculate_overall_p1_score_from_match_matrix(match_matrix, test_langs, dataset)
    
def retrive_p1_by_langs(matrix, all_langs, tgt_langs):
    assert matrix.shape[0] == len(all_langs)
    tgt_idx = [all_langs.index(lang) for lang in tgt_langs]
    return matrix[tgt_idx, :]


""" &&&& Some functions using p1 score for further analysis or illstruation.
1. Cluster languages by relation-wise p1 score. The cluster algorithm can be Kmeans or hierarchical clustering
2. Draw relation-wise p1 score distributions for all languages
"""
def cluster_langs_by_relwise_p1(p1, method='kmeans',clusters=5, save_root=None):
    """Cluster languages based on their object matching counts, then visualize the clustering by 2D graph
    """
    langs = []
    full_match = []
    partial_match = []
    for lang, p1_by_obj in p1.items():
        langs.append(lang)
        full_match.append([p1_by_obj["full-match-per-rel"][rel] for rel in sorted(list(p1_by_obj["full-match-per-rel"].keys()))])
        partial_match.append([p1_by_obj["partial-match-per-rel"][rel] for rel in sorted(list(p1_by_obj["partial-match-per-rel"].keys()))])
    full_match = np.array(full_match)
    partial_match = np.array(partial_match)

    def cluster(match_cnts: list[list]):
        estimator = KMeans(n_clusters=clusters).fit(np.array(match_cnts))
        label_pred = estimator.labels_ 

        return label_pred
    
    def hierarchy_cluster(match_cnts: list[list]):
        from sklearn.cluster import AgglomerativeClustering
        hierarchical_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=1, affinity='euclidean', linkage='ward')
        label_pred = hierarchical_cluster.fit_predict(match_cnts) # type: ignore
        return label_pred
        
    def dim_red(match_cnts: list[list]):
        tsne = TSNE()
        tsne.fit_transform(full_match) # type: ignore
        return tsne.embedding_
    
    def visualize(label_pred, match_cnts_2d, path):
        fig, ax = plt.subplots(figsize=(8, 6))

        for idx in range(clusters):
            ids = np.where(label_pred==idx)[0]
            x_i = match_cnts_2d[ids] # type: ignore
            ax.scatter(x_i[:, 0], x_i[:, 1], label=f"cluster{idx}") # type: ignore
            texts = [
                plt.text(
                    x_i[i][0], x_i[i][1], Language.get(langs[_id]).display_name()
                )
                for i, _id in enumerate(ids)
            ]
            adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'), only_move={"points": "xy", "text": "xy"})
        ax.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
        plt.title("Clustering languages by factual knowledge prediction performance")
        if save_root is not None:
            plt.savefig(os.path.join(save_root, path))
        else:
            plt.show()
    
    if method == "kmeans":
        full_label_pred = cluster(full_match) # type: ignore
    elif method == "hierarchy":
        full_label_pred = hierarchy_cluster(full_match) # type: ignore
    else:
        raise NotImplementedError(f"Cluster method {method} is not supported")
    
    full_match_cnts_2d = dim_red(full_match) # type: ignore
    visualize(full_label_pred, full_match_cnts_2d, "cluster-based-full-match-counts.png")
    
    # partial_label_pred = cluster(partial_match) # type: ignore
    # partial_match_cnts_2d = dim_red(partial_match) # type: ignore
    # visualize(partial_label_pred, partial_match_cnts_2d, "cluster-based-partial-match-counts.png")

def draw_p1_distribution_per_lang(p1, dataset, step=10, save_path=None, reload=False):
    def draw_subplot(rel_list, sorted_langs, rel_p1_per_lang, step):
        x = np.array(list(range(len(rel_list))))
        fig, axs = plt.subplots(len(range(1, len(sorted_langs), step)), figsize=(max(x) / 2.5, 20))
        def plot_partial_distri(axs_, langs):
            axs_.set_xticks(x)
            axs_.plot(
                x,
                np.array(rel_p1_per_lang["en"]),
                color="red",
                marker="o",
                markerfacecolor="black",
                linewidth=1,
                markersize=4,
            )
            for lang in langs:
                if lang != 'en':
                    axs_.plot(x, np.array(rel_p1_per_lang[lang]))
            axs_.legend([dataset.display_lang(lang) for lang in ["en"] + langs], ncol=2, bbox_to_anchor=(1.0, 0.5), loc='center left')

        for i, idx in enumerate(range(1, len(sorted_langs), step)):
            plot_partial_distri(axs[i], sorted_langs[idx : idx + step])
        
        axs[-1].set_xticklabels(rel_list, fontsize=9, rotation=80)
        fig.suptitle("P1 scores for different relations per language", fontsize=30)
        plt.subplots_adjust(hspace=0.15)
        plt.title("Distribution of p1 score per relations for a subset of languages")
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()    
        else:
            plt.show()
    
    def draw_single(rel_list, sorted_langs, rel_p1_per_lang):
        fig, ax = plt.subplots(figsize=(8,6))
        x = np.array(list(range(len(rel_list))))
        ax.plot(
                x,
                np.array(rel_p1_per_lang["en"]),
                color="red",
                marker="o",
                markerfacecolor="black",
                linewidth=1,
                markersize=4,
            )
        for idx, lang in enumerate(sorted_langs):
            if lang != 'en':
                ax.plot(x, np.array(rel_p1_per_lang[lang]))
        ax.set_xticks(x)
        ax.set_xticklabels(rel_list, fontsize=8, rotation=90)
        # fig.suptitle("P1 scores for different relations per language", fontsize=12)
        plt.title("P1 scores for different relations per language", fontsize=12)
        ax.legend([dataset.display_lang(lang) for lang in sorted_langs], ncol=1, bbox_to_anchor=(1.0, 0.5), loc='center left')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    en_idx = dataset.langs.index('en')
    langs_dist_en = language_distance_matrix(dataset.langs)[en_idx]
    sorted_langs = []
    for lang_idx in np.argsort(langs_dist_en):
        if dataset.langs[lang_idx] in p1.keys():
            sorted_langs.append(dataset.langs[lang_idx])
    
    
    rel_p1_per_lang = {}
    for lang in p1.keys():
        p1_per_rel = []
        for rel in dataset.rels:
            rel2p1 = p1[lang]['full-match-per-rel']
            if rel in rel2p1:
                p1_per_rel.append(rel2p1[rel])
            else:
                p1_per_rel.append(0)
        rel_p1_per_lang[lang] = np.array(p1_per_rel)

    sorted_rels_idx = np.argsort(rel_p1_per_lang['en'])[::-1]
    rel_list = [dataset.display_rel(dataset.rels[rel_idx]) for rel_idx in sorted_rels_idx]
    for lang in rel_p1_per_lang:
        rel_p1_per_lang[lang] = rel_p1_per_lang[lang][sorted_rels_idx]

    if step >= len(p1):
        draw_single(rel_list, sorted_langs, rel_p1_per_lang)
    else:
        draw_subplot(rel_list, sorted_langs, rel_p1_per_lang, step)

def draw_p1_distribution_part_lang(p1, langs, dataset, step=10, save_path=None, reload=False):
    p1_sub_lang = {k: p1[k] for k in langs}
    draw_p1_distribution_per_lang(p1_sub_lang, dataset, step=step, reload=True)


"""Following up the above discussion of language distance. 
The relation-wise p1 score is not good enough for measure distance between languages. 
The uuid-level matching count can be a better metric for measuring language distances. 
"""
def full_match_uuids(gold, evl):
    matched_uuid = []
    df = pd.DataFrame()
    for idx in gold.index:
        gold_row = gold.loc[idx]
        obj_ids = gold_row["obj_ids"] if isinstance(gold_row["obj_ids"], list) else ast.literal_eval(gold_row["obj_ids"])
        obj_ids = strip_space(obj_ids, is_wrapped=False)
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = evl[evl["id"] == idx]
        for index, row in preds.iterrows():
            pred_list = row.pred_ids if isinstance(row.pred_ids, list) else ast.literal_eval(row.pred_ids)
            pred_list = strip_space(pred_list)
            if obj_ids == pred_list:
                matched_uuid.append(gold_row["uuid"])
    return matched_uuid

def _get_uuid_matching_info(lang, rel):
    gold = dataset_for_mlthread.get_lang_type(lang, rel)
    evl = read_pred(lang, rel, PREDICTION_ROOT[dataset_for_mlthread.model_name])
    return lang, rel, gold["uuid"].tolist(), full_match_uuids(gold, evl)

@loader
def get_all_and_matched_uuid_lsts(dataset, reload=False):
    global dataset_for_mlthread
    dataset_for_mlthread = dataset
    results = []
    with ProcessPoolExecutor(56) as e:
        fut = [e.submit(_get_uuid_matching_info, lang, rel) for lang, rel in dataset.lang_rel_iter()]
        results.extend(r.result() for r in tqdm(as_completed(fut), total=len(fut), desc="Collecting all and matched uuids for language distance measurement"))

    lang2rel2all_uuid = {}
    lang2rel2matched_uuid = {}
    for lang, rel, all_uuids, matched_uuids in results:
        if lang not in lang2rel2all_uuid:
            lang2rel2all_uuid[lang] = {rel: all_uuids}
            lang2rel2matched_uuid[lang] = {rel: matched_uuids}
        else:
            lang2rel2all_uuid[lang].update({rel: all_uuids})
            lang2rel2matched_uuid[lang].update({rel: matched_uuids})

    return lang2rel2all_uuid, lang2rel2matched_uuid

@loader
def calculate_langsim_by_objectwise_p1_with_rel(dataset, reload=False):
    langs = dataset.langs
    rels = dataset.rels
    dists = np.zeros((len(rels), len(langs), len(langs)))
    lang2rel2all_uuid, lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset)
    for lang1, lang2 in tqdm(list(itertools.combinations(langs, 2)), desc="Calculating distances between languages based on matched uuids"):
        idx1 = langs.index(lang1)
        idx2 = langs.index(lang2)
        for rel_ind, rel in enumerate(rels):
            if rel in lang2rel2matched_uuid[lang1] and rel in lang2rel2matched_uuid[lang2]:
                all_intersect = set(lang2rel2all_uuid[lang1][rel]).intersection(lang2rel2all_uuid[lang2][rel])
                lang1_matched = set(all_intersect).intersection(lang2rel2matched_uuid[lang1][rel])
                lang2_matched = set(all_intersect).intersection(lang2rel2matched_uuid[lang2][rel])
                intersect_matched_cnt = len(lang1_matched.intersection(lang2_matched))
                dists[rel_ind][idx1][idx2] = round(intersect_matched_cnt/len(lang1_matched), 6) if len(lang1_matched)>0 else 0
                dists[rel_ind][idx2][idx1] = round(intersect_matched_cnt/len(lang2_matched), 6) if len(lang2_matched)>0 else 0
            else:
                dists[rel_ind][idx1][idx2] = np.NaN
                dists[rel_ind][idx2][idx1] = np.NaN
            np.fill_diagonal(dists[rel_ind], 1)
    return langs, rels, dists

@loader
def calculate_langsim_by_objectwise_p1_without_rel(dataset, reload=False):
    langs = dataset.langs
    rels = dataset.rels
    dists = np.zeros((len(langs), len(langs)))
    lang2rel2all_uuid, lang2rel2matched_uuid = get_all_and_matched_uuid_lsts(dataset)
    for lang1, lang2 in tqdm(list(itertools.combinations(langs, 2)), desc="Calculating distances between languages based on matched uuids"):
        idx1 = langs.index(lang1)
        idx2 = langs.index(lang2)
        # intersect_all = set()
        lang1_matched = set()
        lang2_matched = set()
        for rel in rels:
            if rel in lang2rel2matched_uuid[lang1] and rel in lang2rel2matched_uuid[lang2]:
                intersect_by_rel = set(lang2rel2all_uuid[lang1][rel]).intersection(lang2rel2all_uuid[lang2][rel])
                lang1_matched = lang1_matched.union(intersect_by_rel.intersection(set(lang2rel2matched_uuid[lang1][rel])))
                lang2_matched = lang2_matched.union(intersect_by_rel.intersection(set(lang2rel2matched_uuid[lang2][rel])))
        intersect_matched = lang1_matched.intersection(lang2_matched)
        union_matched = lang1_matched.union(lang2_matched)
        dists[idx1][idx2] = round(len(intersect_matched)/len(union_matched), 6) if len(union_matched)>0 else 0
        dists[idx2][idx1] = round(len(intersect_matched)/len(union_matched), 6) if len(union_matched)>0 else 0
        np.fill_diagonal(dists, 1)
    return langs, rels, dists

"""Clustering languages and drawing heatmap based on the language distance.
"""
def cluster_languages_by_p1_distance(langs, dists):
    # Cluster languages based on the pairwise language distances
    import kmedoids
    clusters = 8
    c = kmedoids.fasterpam(1 - dists, clusters)
    groups = []
    for i in range(clusters):
        index = np.where(c.labels==i)[0].tolist()
        print(" ".join([dataset.display_lang(langs[idx]) for idx in index]))
        groups.append(index)

def draw_heatmap_for_pairwise_langsim_based_on_uuid_matching_rate(dataset: MaskedDataset, langs: list[str], dists, sorted=True):
    if sorted:
        reidx = np.argsort(dists[langs.index('en')])[::-1]
        langs = [langs[idx] for idx in reidx]
        dists = dists[reidx, :][:, reidx]

    import matplotlib.pyplot as plt
    import seaborn as sns

    np.fill_diagonal(dists, np.nan)
    fig, ax = plt.subplots(figsize=(9,6))
    langs_name = [dataset.display_lang(lang, prefix=False) for lang in langs]
    g = sns.heatmap(dists, cmap="coolwarm", xticklabels=langs_name, yticklabels=langs_name, ax=ax) # type: ignore
    
    ax.set_xticklabels(langs_name, fontsize=14)
    ax.set_yticklabels(langs_name, fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)  # set the tick labels font size
    cbar.set_label('Jaccard Distance', size=14)  # set the colorbar label font size
    # plt.title("Distance of languages measured by shared factual knowledge", fontsize=18)
    ax.set_title("Distance of languages measured by shared factual knowledge", y=1.05, fontsize=17)

"""Evaluate the p1 score by token counts of objects: object with single token VS object with multiple tokens
"""
def single_multi_token_match(gold, evl):
    single_token_objs_cnt = 0
    single_token_match_cnt = 0
    multi_tokens_objs_cnt = 0
    multi_tokens_match_cnt = 0
    for idx in gold.index:
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        obj_ids = gold.loc[idx]["obj_ids"] if isinstance(gold.loc[idx]["obj_ids"], list) else ast.literal_eval(gold.loc[idx]["obj_ids"])
        obj_ids = strip_space(obj_ids, is_wrapped=False)
        preds = []
        for ids in evl[evl["id"] == idx]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append(strip_space(ids, is_wrapped=False))

        if len(obj_ids) == 1:
            single_token_objs_cnt += 1
            if obj_ids in preds:
                single_token_match_cnt += 1
        else:
            multi_tokens_objs_cnt += 1
            if obj_ids in preds:
                multi_tokens_match_cnt += 1

    return (
        single_token_match_cnt,
        single_token_objs_cnt,
        multi_tokens_match_cnt,
        multi_tokens_objs_cnt,
    )

def single_multi_token_match_p1(dataset, lang):
    all_single_token_match_cnt = 0
    all_multi_tokens_match_cnt = 0
    all_single_token_objs_cnt = 0
    all_multi_tokens_objs_cnt = 0
    single_token_match_per_rel = {}
    multi_token_match_per_rel = {}
    token_objs_per_rel = {}
    for rel in dataset.get_rels_in_lang(lang):
        try:
            gold = dataset.get_lang_type(lang, rel)
            evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
            (
                single_token_match_cnt,
                single_token_objs_cnt,
                multi_tokens_match_cnt,
                multi_tokens_objs_cnt,
            ) = single_multi_token_match(gold, evl)
            if single_token_objs_cnt != 0:
                single_token_match_per_rel[rel] = round(single_token_match_cnt / single_token_objs_cnt)
            else:
                single_token_match_per_rel[rel] = 0
            if multi_tokens_objs_cnt != 0:
                multi_token_match_per_rel[rel] = round(multi_tokens_match_cnt / multi_tokens_objs_cnt)
            else:
                multi_token_match_per_rel[rel] = 0
            
            all_single_token_match_cnt += single_token_match_cnt
            all_single_token_objs_cnt += single_token_objs_cnt
            all_multi_tokens_match_cnt += multi_tokens_match_cnt
            all_multi_tokens_objs_cnt += multi_tokens_objs_cnt            
            token_objs_per_rel[rel] = {"single_token_objs_cnt": single_token_objs_cnt, "multi_tokens_objs_cnt": multi_tokens_objs_cnt}
        except Exception:
            print(f"Error occurs for {lang}-{rel}")
    single_p1 = round(all_single_token_match_cnt / all_single_token_objs_cnt, 6)
    multi_p1 = round(all_multi_tokens_match_cnt / all_multi_tokens_objs_cnt, 6)
    # print(f"The single token prediction p1 score for {lang} is {single_p1}")
    # print(f"The multiple tokens prediction for {lang} is {multi_p1}\n")
    return {
        lang: {
            "singe-token": single_p1, 
            "multi-tokens": multi_p1,
            "single_token_per_rel": single_token_match_per_rel,
            "multi_token_per_rel": single_token_match_per_rel,
            "token_cnt_per_rel": token_objs_per_rel
            }}

@loader
def single_multi_evaluate_parallel(dataset, thread_num=56, reload=False):
    # sourcery skip: dict-assign-update-to-union
    p1 = {}
    with ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [
            executor.submit(single_multi_token_match_p1, dataset, lang)
            for lang in dataset.langs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running single_multi_evaluate_parallel for model {dataset.model_name}"):
            p1.update(future.result())
    return p1

"""Evaluate object distributions for correct and wrong predictions
Then, draw object distribution for all languages per relations. 
The graph could have very long x-axis as it takes all objects in each relation as x-axis.
"""
def full_match_distri(gold, evl):
    """
    Language-Relation-level function: return counter of matched and unmatched prediction for all obj_uri
    Input:
    - gold, evl: gold data and prediction give language and relations
    Return:
    - match_counter: {"Q150": 5, ...}
    - no_match_counter: {"Q150": 12, ...}
    """
    match_counter = Counter()
    no_match_counter = Counter()
    for idx in gold.index:
        obj_uri = gold.loc[idx]["obj_uri"]
        obj_ids = gold.loc[idx]["obj_ids"] if isinstance(gold.loc[idx]["obj_ids"], list) else ast.literal_eval(gold.loc[idx]["obj_ids"])
        obj_ids = strip_space(obj_ids)
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = []
        for ids in evl[evl["id"] == idx]["pred_ids"].tolist():
            if isinstance(ids, str):
               ids = ast.literal_eval(ids)
            preds.append(strip_space(ids, is_wrapped=False))
        if obj_ids in preds:
            match_counter[obj_uri] += 1
        else:
            no_match_counter[obj_uri] += 1
    return match_counter, no_match_counter

def partial_match_distri(gold, evl):
    match_counter = Counter()
    no_match_counter = Counter()
    for idx in gold.index:
        matched = False
        obj_uri = gold.loc[idx]["obj_uri"]
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]["obj_ids"]))
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = [
            strip_space(ast.literal_eval(ids_str))
            for ids_str in evl[evl["id"] == idx]["pred_ids"].tolist()
        ]
        for pred_ids in preds:
            if all(x in pred_ids for x in obj_ids):
                match_counter[obj_uri] += 1
                matched = True
                break
        if not matched:
            no_match_counter[obj_uri] += 1
    return match_counter, no_match_counter

@loader
def evaluate_obj_distribution(dataset, match_type="full-match", reload=False):
    """
    Return the distribution of object prediction
    1. Count (un)matched objects in all languages for all relations
    2. Reconstruct the counter data instance into the form of:
        {'P131':
            {
                'Q869': {'ms': 1, 'nl': 1, 'de': 1, 'id': 1},
                'Q227': {'ms': 1, 'id': 1, 'tr': 1},
                ...
            }}
    """
    print(f"Start to evaluate distribution of object prediction by {match_type}")
    all_match_counter = {}
    all_no_match_counter = {}
    for lang in tqdm(dataset.langs, desc="Evaluating objects distri across languages"):
        for rel in dataset.get_rels_in_lang(lang):
            gold = dataset.get_lang_type(lang, rel)
            evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
            if match_type == "full-match":
                match_counter, no_match_counter = full_match_distri(gold, evl)
            elif match_type == "partial-match":
                match_counter, no_match_counter = partial_match_distri(gold, evl)
            else:
                raise Exception(f"Undefined matching type - {match_type}")
            if rel not in all_match_counter:
                all_match_counter[rel] = {lang: match_counter}
                all_no_match_counter[rel] = {lang: no_match_counter}
            else:
                all_match_counter[rel].update({lang: match_counter})
                all_no_match_counter[rel].update({lang: no_match_counter})

    all_matched_uri_infos = {}
    all_no_matched_uri_infos = {}
    for rel in all_match_counter:
        matched_uri_infos = {}
        no_matched_uri_infos = {}
        for lang in all_match_counter[rel].keys():
            match_counter = all_match_counter[rel][lang]
            for obj_uri in match_counter:
                if obj_uri not in matched_uri_infos:
                    matched_uri_infos[obj_uri] = {lang: match_counter[obj_uri]}
                else:
                    matched_uri_infos[obj_uri].update({lang: match_counter[obj_uri]})
            no_match_counter = all_no_match_counter[rel][lang]
            for obj_uri in no_match_counter:
                if obj_uri not in no_matched_uri_infos:
                    no_matched_uri_infos[obj_uri] = {lang: no_match_counter[obj_uri]}
                else:
                    no_matched_uri_infos[obj_uri].update(
                        {lang: no_match_counter[obj_uri]}
                    )
        all_matched_uri_infos[rel] = matched_uri_infos
        all_no_matched_uri_infos[rel] = no_matched_uri_infos
    return all_matched_uri_infos, all_no_matched_uri_infos

def get_x_y_obj_distri(dataset, matching_infos):
    """Generate x, y axis data for draw distribution for mask prediction accuracy in different languages 
    Returns:
    rel2objtoken
    distributions
    """
    rel2objs = {}
    for rel in matching_infos.keys():
        uris = sorted(list(matching_infos[rel].keys()))
        rel2objs[rel] = uris

    distributions = {}
    for lang in dataset.langs:
        rel2distri = {}
        distri = []
        for rel in matching_infos.keys():
            distri = []
            for obj_uri in rel2objs[rel]:
                if lang in matching_infos[rel][obj_uri]:
                    distri.append(matching_infos[rel][obj_uri][lang])
                else:
                    distri.append(0)
            rel2distri[rel] = distri
        distributions[lang] = rel2distri

    rel2objtoken = {
        rel: [dataset.display_name(uri) for uri in rel2objs[rel]]
        for rel in rel2objs
    }
    return rel2objtoken, distributions

def draw_obj_distribution(
    obj_list, distri, sorted_langs, rel, model_name, name, reload=False
):
    root = os.path.join(EVALUATION_ROOT[model_name], "objects-distribution", rel)
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, name)
    if os.path.exists(path) and reload is False:
        print("{} is generated".format(path))
        return
    step = 10
    x = np.array(list(range(len(obj_list))))

    try:
        fig, axs = plt.subplots(
            len(range(1, len(sorted_langs), step)), figsize=(max(x) / 2.5, 20)
        )

        def plot_partial_distri(axs_, langs):
            axs_.set_xticks(x, obj_list)
            axs_.set_xticklabels(obj_list, fontsize=8, rotation=50)
            axs_.plot(
                x,
                np.array(distri["en"][rel]),
                color="red",
                marker="o",
                markerfacecolor="black",
                linewidth=1,
                markersize=4,
            )
            for lang in langs:
                axs_.plot(x, np.array(distri[lang][rel]))
            axs_.legend(["en"] + langs, ncol=2)

        for i, idx in enumerate(range(1, len(sorted_langs), step)):
            plot_partial_distri(axs[i], sorted_langs[idx : idx + step])

        plt.subplots_adjust(hspace=0.3)
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print("Failed to draw relation of {}".format(rel))
    return

def draw_all_obj_distributions(dataset):
    def draw(dataset, matched_infos, no_matched_infos, name_format_prefix, reload):
        matched_rel2objs, matched_distri = get_x_y_obj_distri(dataset, matched_infos)
        no_matched_rel2objs, no_matched_distri = get_x_y_obj_distri(
            dataset, no_matched_infos
        )

        lang_dists = language_distance(dataset.langs)
        dist_type_to_sorted_lang_dist = {}
        for dist_name in lang_dists.keys():
            dist_type_to_sorted_lang_dist[dist_name] = []
            for lang_idx in np.argsort(lang_dists[dist_name]):
                dist_type_to_sorted_lang_dist[dist_name].append(dataset.langs[lang_idx])
        for rel in tqdm(matched_rel2objs.keys(), desc="Drawing obj distri for all relations"):
            for dist_type in lang_dists.keys():
                obj_list = matched_rel2objs[rel]
                draw_obj_distribution(
                    obj_list,
                    matched_distri,
                    dist_type_to_sorted_lang_dist[dist_type],
                    rel,
                    dataset.model_name,
                    "{}-matched-sorted-by-{}".format(name_format_prefix, dist_type),
                    reload,
                )
                obj_list = no_matched_rel2objs[rel]
                draw_obj_distribution(
                    obj_list,
                    no_matched_distri,
                    dist_type_to_sorted_lang_dist[dist_type],
                    rel,
                    dataset.model_name,
                    "{}-no-matched-sorted-by-{}".format(name_format_prefix, dist_type),
                    reload,
                )

    print("Evaluate object distribution predicted by full-match")
    matched_infos, no_matched_infos = evaluate_obj_distribution(
        dataset, match_type="full-match", reload=False
    )
    draw(
        dataset,
        matched_infos,
        no_matched_infos,
        dataset.model_name + "-full",
        reload=True,
    )

    print("Evaluate object distribution predicted by partial-match")
    matched_infos, no_matched_infos = evaluate_obj_distribution(
        dataset, match_type="partial-match", reload=True
    )
    draw(
        dataset,
        matched_infos,
        no_matched_infos,
        dataset.model_name + "-partial",
        reload=True,
    )


class Evaluator:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def evaluate_p1_score(self, thread_num=56):
        return p1_evaluate_parallel(dataset=self.dataset, thread_num=thread_num)

    def evaluate_single_multi_token_p1(self, thread_num=56):
        return single_multi_evaluate_parallel(
            dataset=self.dataset, thread_num=thread_num
        )

    def draw_obj_distributions(self):
        draw_all_obj_distributions(dataset=self.dataset)


if __name__ == "__main__":
    from constants import MODELS

    # mlama = MaskedDataset(reload=False)
    # distance = language_distance(mlama.langs)
    # for model_name in MODELS:
    #     mlama = MaskedDataset(model_name=model_name)
    #     p1 = p1_evaluate_parallel(dataset=mlama, thread_num=56)
    #     p2 = single_multi_evaluate_parallel(dataset=mlama, thread_num=56)

    ## Draw object distributions
    # from mask_dataset import MaskedDataset
    # for model_name in MODELS:
    #     dataset = MaskedDataset(model_name=model_name)
    #     draw_all_obj_distributions(dataset=dataset)

    dataset = MaskedDataset(model_name="xlmr")
    # langs = dataset.langs
    xlmr_langs, xlmr_uuids, xlmr_full_matrix = get_full_match_matrix_by_uuid(dataset, reload=True)
    # langs.remove("en")
    # _get_subject_object_cooccurence_in_article(
    #     dataset=dataset,
    #     candidate_langs=langs,
    #     reload=False)
    # plot_stacted_bar_and_percentage_of_fk_matching(
    #     dataset=dataset, 
    #     resource_type="article", 
    #     candidate_langs=['ca', 'da', 'fi', 'ga', 'he', 'ja', 'ka', 'ko', 'ms', 'nl', 'ru', 'sr', 'th', 'zh', 'it'])
    # generate_match_sentences(dataset)
    
    
    # _get_wiki_matches_resource_from_dumped_wiki_abstract(dataset, reload=True)
    # _get_subject_object_cooccurence_in_abstract(dataset, reload=True)
    # get_wiki_matches_matrix_from_dumped_wiki_abstract(dataset, reload=True)

    # _get_title_object_subject_matchings(dataset, reload=True)
    # get_wiki_matches_matrix_from_dumped_wiki_title(dataset, reload=False)