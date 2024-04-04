import os
import ast
import pandas as pd
from transformers import AutoTokenizer, BertTokenizer
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from utils import load_objects, parse_list, tokens2id

XLMR_PREDICTION_ROOT = "../result/prediction-xlmr"
MBERT_PREDICTION_ROOT = "../result/prediction-mbert"

PREDICTION_ROOT = {
    "xlmr": XLMR_PREDICTION_ROOT, 
    "mbert": MBERT_PREDICTION_ROOT
}

def read_pred(lang, rel, root):
    tgt_fn = os.path.join(root, lang, "{}-{}.csv".format(lang, rel))
    if not os.path.exists(tgt_fn):
        return pd.DataFrame()
    return pd.read_csv(tgt_fn, index_col=0)


# Evaluate p1 score by two methods: full match and partial match.
def strip_match(gold, evl):
    match_cnt = 0
    matches = []
    for idx in gold.index:
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]['obj_ids']))
        preds = [strip_space(ast.literal_eval(ids_str)) 
                 for ids_str in evl[evl['id']==idx]['pred_ids'].tolist()]
        if obj_ids in preds:
            matches.append(idx)
            match_cnt += 1
    return match_cnt, matches

def partial_match(gold, evl):
    match_cnt = 0
    matches = []
    for idx in gold.index:
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]['obj_ids']))
        preds = [strip_space(ast.literal_eval(ids_str)) 
                 for ids_str in evl[evl['id']==idx]['pred_ids'].tolist()]
        for pred_ids in preds:
            if all(x in pred_ids for x in obj_ids):
                matches.append(idx)
                match_cnt += 1
                break
    return match_cnt, matches

def evaluate_p1_by_language(dataset, model_name, lang):
    objs = load_objects(lang, model_name, None)
    query_cnt = 0
    strip_match_cnt = 0
    partial_match_cnt = 0
    for rel in objs.keys():
        try:
            gold = dataset.get_lang_type(lang, rel)
            evl = read_pred(lang, rel, PREDICTION_ROOT[model_name])
            strip, strip_matches = strip_match(gold, evl)
            partial, partial_matches = partial_match(gold, evl)
            strip_match_cnt += strip
            partial_match_cnt += partial
            query_cnt += len(gold)
        except: 
            print("Error occurs for {}-{}".format(lang, rel))
    strip_p1 = round(strip_match_cnt/query_cnt, 6)
    partial_p1 = round(partial_match_cnt/query_cnt, 6)
    # print('The strip match p1 score for {} is {}'.format(lang, strip_p1))
    # print('The parital match p1 score for {} is {}\n'.format(lang, partial_p1))
    return {lang: [strip_p1, partial_p1]}

def p1_evaluate_parallel(dataset, model_name, thread_num):    
    p1 = {}
    with ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(evaluate_p1_by_language, dataset, model_name, lang) for lang in dataset.langs]
        for future in as_completed(futures):
            p1.update(future.result())
    return p1

# Evaluate the p1 score by token counts of objects: object with single token VS object with multiple tokens
def single_multi_token_match(gold, evl):
    match_cnt = 0
    single_token_objs_cnt = 0
    single_token_match_cnt = 0
    # single_token_matches = []
    multi_tokens_objs_cnt = 0
    multi_tokens_match_cnt = 0
    # multi_tokens_matches = []
    for idx in gold.index:
        preds = [strip_space(ast.literal_eval(ids_str)) 
                     for ids_str in evl[evl['id']==idx]['pred_ids'].tolist()]
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]['obj_ids']))
        if len(obj_ids) == 1:
            single_token_objs_cnt += 1
            if obj_ids in preds:
                # single_token_matches.append(idx)
                single_token_match_cnt += 1
        else:
            multi_tokens_objs_cnt += 1
            if obj_ids in preds:
                # multi_tokens_matches.append(idx)
                multi_tokens_match_cnt += 1
            
            
    return single_token_match_cnt, single_token_objs_cnt, multi_tokens_match_cnt, multi_tokens_objs_cnt

def single_multi_token_match_p1(dataset, model_name, lang):
    objs = load_objects(lang, model_name, None)
    query_cnt = 0
    all_single_token_match_cnt = 0
    all_multi_tokens_match_cnt = 0
    all_single_token_objs_cnt = 0
    all_multi_tokens_objs_cnt = 0
    for rel in objs.keys():
        try:
            gold = dataset.get_lang_type(lang, rel)
            evl = read_pred(lang, rel, PREDICTION_ROOT[model_name])
            single_token_match_cnt, single_token_objs_cnt, multi_tokens_match_cnt, multi_tokens_objs_cnt = single_multi_token_match(gold, evl)
            all_single_token_match_cnt += single_token_match_cnt
            all_single_token_objs_cnt += single_token_objs_cnt
            all_multi_tokens_match_cnt += multi_tokens_match_cnt
            all_multi_tokens_objs_cnt += multi_tokens_objs_cnt
        except: 
            print("Error occurs for {}-{}".format(lang, rel))
    single_p1 = round(all_single_token_match_cnt/all_single_token_objs_cnt, 6)
    multi_p1 = round(all_multi_tokens_match_cnt/all_multi_tokens_objs_cnt, 6)
    print('The single token prediction p1 score for {} is {}'.format(lang, single_p1))
    print('The multiple tokens prediction for {} is {}\n'.format(lang, multi_p1))
    return {lang: [single_p1, multi_p1]}            

def single_multi_evaluate_parallel(dataset, model_name, thread_num):    
    p1 = {}
    with ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(single_multi_token_match_p1, dataset, model_name, lang) for lang in dataset.langs]
        for future in as_completed(futures):
            p1.update(future.result())
    return p1


# Evaluate object distributions for correct and wrong predictions

def evaluate_correct_distribution(gold, evl):
    match_cnt = 0
    matches = []
    for idx in gold.index:
        obj_ids = strip_space(ast.literal_eval(gold.loc[idx]['obj_ids']))
        preds = [strip_space(ast.literal_eval(ids_str)) 
                 for ids_str in evl[evl['id']==idx]['pred_ids'].tolist()]
        for pred_ids in preds:
            if all(x in pred_ids for x in obj_ids):
                matches.append(idx)
                match_cnt += 1
                break
    return match_cnt, matches
    

def evaluate_obj_distribution(dataset, model_name, lang, rel):
    gold = dataset.get_lang_type(lang, rel)
    evl = read_pred(lang, rel, PREDICTION_ROOT[model_name])
    evaluate_correct_distribution(gold, evl)

    
if __name__ == "__main__":
    # convert_old2new('es', 'P264')
    # Convert old prediction csv to clear version
    
    # for lang in mlama.langs:
    #     objs = load_objects(lang, "mbert", None)
    #     print("Start to convert {}".format(lang))
    #     for rel in tqdm(objs.keys()):
    #         convert_old2new(lang, rel)
    
    # evaluate_by_language(mlama, "mbert")
    
    from mask_dataset import MaskedDataset
    model_name = "xlmr"
    mlama = MaskedDataset("mlama", model_name)
    # xlmr_p1 = p1_evaluate_parallel(mlama, model_name, 40)
    xlmr_p1 = single_multi_evaluate_parallel(mlama, model_name, 40)
    
    model_name = "mbert"
    mlama = MaskedDataset("mlama", model_name)
    # mbert_p1 = p1_evaluate_parallel(mlama, model_name, 40)
    mbert_p1 = single_multi_evaluate_parallel(mlama, model_name, 40)
    
    for lang in mlama.langs: 
        print("xlmr p1 accuracy for lang-{}: {}".format(lang, xlmr_p1[lang]))
        print("mbert p1 accuracy for lang-{}: {}\n".format(lang, mbert_p1[lang]))
