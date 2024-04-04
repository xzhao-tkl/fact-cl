import ast
import itertools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import INTERVENTION_ROOT, PREDICTION_ROOT
from mask_dataset import MaskedDataset
from pred_evaluation import get_full_match_cnt_per_obj, get_full_match_per_uuid, read_pred
from utils import get_logger, strip_space

logger = get_logger("neuron_intervention_evaluation.log", __name__)

dataset_for_nie = MaskedDataset()

SUFFIX = "-intervened-"

def get_intervened_evl_df(model_name, rel, base_lang, test_lang, operator, neuron_type="acts"):
    evl = pd.DataFrame()
    candidate_objs = []

    operator_root = os.path.join(INTERVENTION_ROOT[model_name] + SUFFIX + neuron_type, f"obj-{operator}")
    test_lang_root = os.path.join(operator_root, rel, f"{base_lang}-base", f"{test_lang}-test")

    if not os.path.exists(test_lang_root):
        logger.error(f"{test_lang_root} is supposed to exist but not")
        raise FileNotFoundError(f"{test_lang_root} is supposed to exist but not")

    for obj_csv in os.listdir(test_lang_root):
        candidate_objs.append(obj_csv.split('.')[0])
        fn = os.path.join(test_lang_root, obj_csv)
        if os.path.exists(fn):
            evl = pd.concat([evl, pd.read_csv(fn)])
    return evl, candidate_objs

def check_obj2uuid(original_evl, intervened_evl, candiate_objs):
    original_obj2uuids = {}
    intervened_obj2uuids = {}
    for obj in candiate_objs:
        original_obj2uuids[obj] = len(set(original_evl[original_evl['obj_uri']==obj]['uuid'].tolist()))
        intervened_obj2uuids[obj] = len(set(intervened_evl[intervened_evl['obj_uri']==obj]['uuid'].tolist()))
    return original_obj2uuids, intervened_obj2uuids

def get_comparison_group_per_uuid(gold, original_evl, intervened_evl, candiate_objs):
    compare_group = {}
    intervened_uuids = set(intervened_evl['uuid'].tolist())
    all_cnt = 0
    for index, item in original_evl.iterrows():
        if item.obj_uri not in candiate_objs:
            continue
        all_cnt += 1
        if item.uuid not in intervened_uuids:
            obj2gold_uuids = gold[gold['obj_uri']==item.obj_uri]['uuid'].tolist()
            raise ValueError(f"{item.uuid} is not evaluated in intervened models. In gold? : {item.uuid in obj2gold_uuids}")
        mask_nums = intervened_evl[intervened_evl['uuid']==item.uuid]['mask_num'].tolist()
        if item.mask_num not in mask_nums:
            raise ValueError(f"{item.uuid} is evaluated in intervened models but the prompt with {item.mask_num} masks is not evaluated, only has {mask_nums}")
        intervened_item = intervened_evl[(intervened_evl['uuid']==item.uuid) & (intervened_evl['mask_num']==item.mask_num)].iloc[0]
        gold_item = gold[gold['uuid']==item.uuid].iloc[0]
        compare_group.setdefault(item.uuid, []).append({
            "mask_num": item.mask_num,
            "original_pred": (item.prediction, item.pred_ids),
            "intervened_pred": (intervened_item.prediction, intervened_item.pred_ids),
            "gold_answer": (gold_item.obj, gold_item.obj_ids)
        })
    return compare_group

def evaluate_and_compare(compare_group):
    all_cnt = 0
    org_pred_cnt = 0
    itv_pred_cnt = 0
    for uuid in compare_group.keys():
        for pred_item in compare_group[uuid]:
            golden_pred_ids = strip_space(ast.literal_eval(pred_item['gold_answer'][1]), is_wrapped=False)
            original_pred_ids = strip_space(ast.literal_eval(pred_item['original_pred'][1]), is_wrapped=True)
            intervened_pred_ids = strip_space(ast.literal_eval(pred_item['intervened_pred'][1]), is_wrapped=True)
            if golden_pred_ids == original_pred_ids:
                org_pred_cnt += 1
            if golden_pred_ids == intervened_pred_ids:
                itv_pred_cnt += 1
            all_cnt += 1
    return round(org_pred_cnt/all_cnt, 6), round(itv_pred_cnt/all_cnt, 6)

def calcualte_p1(rel, base_lang, test_lang, operator, neuron_type="acts"):
    """Cannot be called by referecing the function in jupyter because the `dataset_for_nie` has to be global variable to fasten calculation
    """
    try:
        original_evl = read_pred(test_lang, rel, PREDICTION_ROOT[dataset_for_nie.model_name])
        gold = dataset_for_nie.get_lang_type(test_lang, rel)
        root = os.path.join(INTERVENTION_ROOT[dataset_for_nie.model_name] + SUFFIX + neuron_type, f"obj-{operator}")
        base_lang_path = os.path.join(root, rel, f"{base_lang}-base")
        if not os.path.exists(base_lang_path):
            return 
        
        intervened_evl, candidate_objs = get_intervened_evl_df(dataset_for_nie.model_name, rel, base_lang, test_lang, operator, neuron_type)
        compare_group = get_comparison_group_per_uuid(gold, original_evl, intervened_evl, candiate_objs=candidate_objs)
        origin_p1, intervened_p1 = evaluate_and_compare(compare_group)
        return origin_p1, intervened_p1, rel, base_lang, test_lang, operator
    except Exception as e:
        logger.error(f"Encounter error for {operator, base_lang, test_lang}, {e}")
        print(f"Encounter error for {operator, base_lang, test_lang}, {e}")
        return None

def calculate_obj_match_count(rel, base_lang, test_lang, operator, neuron_type="acts"):
    """Cannot be called by referecing the function in jupyter because the `dataset_for_nie` has to be global variable to fasten calculation
    """
    try:
        gold = dataset_for_nie.get_lang_type(test_lang, rel)

        root = os.path.join(INTERVENTION_ROOT[dataset_for_nie.model_name] + SUFFIX + neuron_type, f"obj-{operator}")
        base_lang_path = os.path.join(root, rel, f"{base_lang}-base")
        if not os.path.exists(base_lang_path):
            return 
        
        intervened_evl = get_intervened_evl_df(dataset_for_nie.model_name, rel, base_lang, test_lang, operator, neuron_type)[0]
        objs2match_cnt = get_full_match_cnt_per_obj(gold, intervened_evl) 
        return rel, base_lang, test_lang, operator, objs2match_cnt
    except Exception as e:
        return None

def calculate_uuid_match(rel, base_lang, test_lang, operator, neuron_type="acts"):
    """Cannot be called by referecing the function in jupyter because the `dataset_for_nie` has to be global variable to fasten calculation
    """
    try:
        gold = dataset_for_nie.get_lang_type(test_lang, rel)

        root = os.path.join(INTERVENTION_ROOT[dataset_for_nie.model_name] + SUFFIX + neuron_type, f"obj-{operator}")
        base_lang_path = os.path.join(root, rel, f"{base_lang}-base")
        if not os.path.exists(base_lang_path):
            return 
        
        intervened_evl = get_intervened_evl_df(dataset_for_nie.model_name, rel, base_lang, test_lang, operator, neuron_type)[0]
        uuid2match = get_full_match_per_uuid(gold, intervened_evl) 
        return rel, base_lang, test_lang, operator, uuid2match
    except Exception as e:
        return None

def collect_match_matrix_in_parallel(target_rels, base_langs, test_langs, neuron_operators, neuron_type="acts", collection_type="per_obj", reload=False) -> tuple[list[str], list[str], list[str], dict[str, np.ndarray]]:
    if collection_type == "per_obj":
        cahce_root = os.path.join(INTERVENTION_ROOT[dataset_for_nie.model_name] + SUFFIX + neuron_type, 'cache', "per_obj")
        parallelized_func = calculate_obj_match_count
        all_units = dataset_for_nie.get_rel_obj_pairs()
    elif collection_type == "per_uuid":
        cahce_root = os.path.join(INTERVENTION_ROOT[dataset_for_nie.model_name] + SUFFIX + neuron_type, 'cache', "per_uuid")
        parallelized_func = calculate_uuid_match
        all_units = sorted(dataset_for_nie.uuid_info_plain.keys())
    else:
        raise ValueError(f"The collection_type must be either `per_obj` or `per_uuid`, but get {collection_type}")
    
    os.makedirs(cahce_root, exist_ok=True)

    all_base_langs = ["en", "id", "pl", "zh", "pt", "ca", "ms", "sv"]
    for lang in base_langs:
        assert lang in all_base_langs

    all_test_langs = dataset_for_nie.get_sorted_langs()    
    
    check_parameters_fn = os.path.join(cahce_root, 'check_parameters.pkl')
    if os.path.exists(check_parameters_fn):
        with open(check_parameters_fn, 'rb') as fp:
            checked_params = pickle.load(fp)
    else:
        checked_params = set()
    
    operator2match_matrix = {}
    if not reload:
        for neuron_operator in neuron_operators:
            operator_cache_fn = os.path.join(cahce_root, f"{neuron_operator}.pkl")
            if os.path.exists(operator_cache_fn):
                with open(operator_cache_fn, 'rb') as fp:
                    operator2match_matrix[neuron_operator] = pickle.load(fp)
            else:
                operator2match_matrix[neuron_operator] = np.empty((len(all_base_langs), len(all_test_langs), len(all_units)))
                operator2match_matrix[neuron_operator].fill(np.nan)

    rel2objs = {}
    for rel in target_rels:
        rel2objs[rel] = dataset_for_nie.get_objs_in_rel(rel)
    
    need_modification = set()
    with ProcessPoolExecutor(max_workers=45) as executor:
        futures = []
        for operator in tqdm(neuron_operators, desc="Submitting future tasks"):
            for rel in target_rels:
                for base_lang, test_lang in itertools.product(base_langs, test_langs):
                    if (operator, base_lang, test_lang, rel) in checked_params:
                        continue                    
                    need_modification.add(operator)
                    checked_params.add((operator, base_lang, test_lang, rel))
                    futures.append(executor.submit(parallelized_func, rel, base_lang, test_lang, operator, neuron_type))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating p1 score for intervened models"):
            res = future.result()
            if res is not None:
                if collection_type == "per_obj":
                    rel, base_lang, test_lang, operator, objs2match_cnt = res
                    base_idx = all_base_langs.index(base_lang)
                    test_idx = all_test_langs.index(test_lang)    
                    for obj in objs2match_cnt.keys():
                        rel_obj_idx = all_units.index(f"{rel}-{obj}")
                        operator2match_matrix[operator][base_idx][test_idx][rel_obj_idx] = objs2match_cnt[obj]
                else: 
                    rel, base_lang, test_lang, operator, uuid2match = res
                    base_idx = all_base_langs.index(base_lang)
                    test_idx = all_test_langs.index(test_lang)    
                    for uuid in uuid2match.keys():
                        uuid_idx = all_units.index(f"{uuid}")
                        operator2match_matrix[operator][base_idx][test_idx][uuid_idx] = uuid2match[uuid]
                
    for neuron_operator in neuron_operators:
        if neuron_operator not in need_modification:
            continue
        operator_cache_fn = os.path.join(cahce_root, f"{neuron_operator}.pkl")
        with open(operator_cache_fn, 'wb') as fp:
            pickle.dump(operator2match_matrix[neuron_operator], fp)        
    with open(check_parameters_fn, 'wb') as fp:
        pickle.dump(checked_params, fp)
    return all_base_langs, all_test_langs, all_units, operator2match_matrix

def get_objwise_comparison(match_matrix, base_langs, test_langs, neuron_operators, neuron_type="acts"):
    operator2match_matrix = collect_match_matrix_in_parallel(
        target_rels = target_rels, 
        base_langs = base_langs, 
        test_langs = test_langs, 
        neuron_operators = neuron_operators,
        reload=True)

    
if __name__ == "__main__":
    target_rels = dataset_for_nie.rels
    base_langs = ['en', 'id', 'pl', 'zh']
    
    test_langs = [
        'en', 'es', 'de', 'da',
        'id', 'ms', 'vi', 'sv',
        'pl', 'hu', 'sk', 'cs',
        'zh', 'ko', 'sr', 'ja',
        "it", "nl", "pt", "ca", "tr", "fr", "af", "ro", "gl", "fa", "el", "cy"]
    
    neuron_operators = [
        'top5_0', 'top5_1.2', 'top5_1.5', 'top5_1.8', 'top5_2', 'top5_3', 'top5_5',
        'top10_0', 'top10_1.2', 'top10_1.5', 'top10_1.8', 'top10_2', 'top10_3', 'top10_5',
        'top30_0', 'top30_1.2', 'top30_1.5', 'top30_1.8', 'top30_2', 'top30_3', 'top10_5',
        'top50_0', 'top50_1.2', 'top50_1.5', 'top50_1.8', 'top50_2', 'top50_3', 'top10_5',
        'top100_0', 'top100_1.2','top100_1.5','top100_1.8','top100_2','top100_3', 'top10_5']
 

    # base_langs = ['en']
    # test_langs = ['en', 'es', 'de', 'da', 'id', 'ms', 'vi', 'sv', "it", "nl", "pt", "ca", "tr", "fr", "af"]
    # neuron_operators = [
    #     'top5_0', 'top5_1.2', 'top5_1.5', 'top5_1.8', 'top5_2',
    #     'top10_0', 'top10_1.2', 'top10_1.5', 'top10_1.8', 'top10_2',
    #     'top30_0', 'top30_1.2', 'top30_1.5', 'top30_1.8', 'top30_2',
    #     'top50_0', 'top50_1.2', 'top50_1.5', 'top50_1.8', 'top50_2']
    
    # all_base_langs, all_test_langs, all_uuids, operator2match_matrix_outs = collect_match_matrix_in_parallel(
    #     target_rels = target_rels, 
    #     base_langs = base_langs, 
    #     test_langs = test_langs, 
    #     neuron_operators = neuron_operators,
    #     neuron_type = "outs",
    #     collection_type="per_uuid")
    
    all_base_langs, all_test_langs, all_uuids, operator2match_matrix_acts = collect_match_matrix_in_parallel(
        target_rels = target_rels, 
        base_langs = base_langs, 
        test_langs = test_langs, 
        neuron_operators = neuron_operators,
        neuron_type = "acts",
        collection_type="per_uuid")