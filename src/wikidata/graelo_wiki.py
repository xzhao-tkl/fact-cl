from tqdm import tqdm
from datasets import load_dataset
from elasticsearch import Elasticsearch
import sys
sys.path.append('../../src/')

from utils import batchify, loader

langs = ['ms', 'ca', 'ko', 'he', 'fi', 'ga', 'ka', 'en', 'th', 'nl', 
         'zh', 'ja', 'eu', 'da', 'pt', 'ru', 'fr', 'sr', 'et', 'sv', 
         'hy', 'cy', 'sq', 'it', 'hi', 'hr', 'es', 'hu', 'bg', 'ta', 
         'sl', 'bn', 'de', 'id', 'uk', 'be', 'ceb', 'el', 'fa', 'pl', 
         'az', 'ar', 'la', 'gl', 'lt', 'cs', 'sk', 'lv', 'tr', 'af', 
         'vi', 'ur', 'ro']
        
def reindex_graelo_wiki():
    def add_index(es_client, lang):
        wikipedia = load_dataset("graelo/wikipedia", f"20230601.{lang}", split='train', cache_dir="/disk/xzhao/datasets/huggingface")
        if not es_client.indices.exists(index=f"{lang}_title"):
            print(f"Start to index {lang} title")
            wikipedia.add_elasticsearch_index(column="title", es_client=es_client, es_index_name=f"{lang}_title") # type: ignore
        else:
            print(f"Indexing of {lang} title is already done")

    es_client = Elasticsearch("http://localhost:9200")
    for lang in langs:
        add_index(es_client, lang)

def _search_index_by_batch(queries, lang, es_client, search_dataset=None):
    if search_dataset is None:
        search_dataset = load_dataset("graelo/wikipedia", f"20230601.{lang}", split='train', cache_dir="/disk/xzhao/datasets/huggingface")
    
    search_dataset.load_elasticsearch_index('title', es_client=es_client, es_index_name=f"{lang}_title") # type: ignore
    return search_dataset.get_nearest_examples_batch(index_name="title", queries=queries, k=20)[1] # type: ignore

def _search_index_per_query(queries, lang, es_client, search_dataset=None):
    if search_dataset is None:
        search_dataset = load_dataset("graelo/wikipedia", f"20230601.{lang}", split='train', cache_dir="/disk/xzhao/datasets/huggingface")
    
    search_dataset.load_elasticsearch_index('title', es_client=es_client, es_index_name=f"{lang}_title") # type: ignore
    retrieved = []
    new_queries = []
    for query in queries:
        try:
            retrieved.append(search_dataset.get_nearest_examples(index_name="title", query=query, k=20)[1]) # type: ignore
            new_queries.append(query)
        except Exception as e:
            pass
    return new_queries, retrieved

def _load_retraining_text(lang, queries, search_dataset=None, es_client=None):
    def _load_by_batch(lang, queries, search_dataset, es_client):
        matched = 0
        sub2text = {}
        try:
            batched_results = _search_index_by_batch(queries, lang, es_client, search_dataset=search_dataset)
        except Exception as e:
            queries, batched_results = _search_index_per_query(queries, lang, es_client, search_dataset=search_dataset)

        for query, retrieved in zip(queries, batched_results):
            if query in retrieved['title']:
                idx = retrieved['title'].index(query)
                sub2text[query] = {"title": retrieved['title'][idx], "text": retrieved['text'][idx]}
                matched += 1
        unmatched = len(queries) - matched
        return matched, unmatched, sub2text


    if search_dataset is None:
        search_dataset = load_dataset("graelo/wikipedia", f"20230601.{lang}", split='train', cache_dir="/disk/xzhao/datasets/huggingface")
    
    if es_client is None:
        es_client = Elasticsearch("http://localhost:9200")

    matched = 0
    unmatched = 0
    sub2text = {}
    
    for _queries in tqdm(list(batchify(queries, 128)), "Search subject by batch"):
        _matched, _unmatched, _sub2text = _load_by_batch(_queries, lang, search_dataset=search_dataset, es_client=es_client)
        matched += _matched
        unmatched += _unmatched
        sub2text.update(_sub2text)

    return matched, unmatched, sub2text

@loader
def load_retraining_text_by_lang(dataset, lang, es_client=None, reload=False):
    if es_client is None:
        es_client = Elasticsearch("http://localhost:9200")

    lang2subs = {}
    lang2objs = {}
    sub_info = dataset.get_sub_info(reload=False)
    obj_info = dataset.get_obj_info(reload=False)
    for _lang in dataset.langs:
        lang2subs[_lang] = []
        lang2objs[_lang] = []
        for lang2sub_info in sub_info.values():
            if _lang in lang2sub_info:
                lang2subs[_lang].append(lang2sub_info[_lang]["sub"])
        for lang2obj_info in obj_info.values():
            if _lang in lang2obj_info:
                lang2objs[_lang].append(lang2obj_info[_lang]["obj"])

    queries = lang2subs[lang]
    wikipedia = load_dataset("graelo/wikipedia", f"20230601.{lang}", split='train', cache_dir="/disk/xzhao/datasets/huggingface")    
    matched_sub, unmatched_sub, sub2text = _load_retraining_text(queries, lang, search_dataset=wikipedia, es_client=es_client)
    queries = lang2objs[lang]
    matched_obj, unmatched_obj, obj2text = _load_retraining_text(queries, lang, search_dataset=wikipedia, es_client=es_client)
    return matched_sub, unmatched_sub, sub2text, matched_obj, unmatched_obj, obj2text

def load_retraining_text_all_langs(dataset, lang):
    lang2all = {}
    for lang in dataset.langs:
        matched_sub, unmatched_sub, sub2text, matched_obj, unmatched_obj, obj2text = load_retraining_text_by_lang(dataset, lang)
        lang2all[lang] = (matched_sub, unmatched_sub, sub2text, matched_obj, unmatched_obj, obj2text)
    return lang2all
