import gzip
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from urllib import request

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append('../../src/')
from constants import WIKI_URI_ROOT
from mask_dataset import MaskedDataset

langs = ['ms', 'ca', 'ko', 'he', 'fi', 'ga', 'ka', 'en', 'th', 'nl', 
        'zh', 'ja', 'eu', 'da', 'pt', 'ru', 'fr', 'sr', 'et', 'sv', 
        'hy', 'cy', 'sq', 'it', 'hi', 'hr', 'es', 'hu', 'bg', 'ta', 
        'sl', 'bn', 'de', 'id', 'uk', 'be', 'ceb', 'el', 'fa', 'pl', 
        'az', 'ar', 'la', 'gl', 'lt', 'cs', 'sk', 'lv', 'tr', 'af', 
        'vi', 'ur', 'ro']


"""Get urls of wikidata page
"""
def locate_urls(reload=False):
    file_path = os.path.join(WIKI_URI_ROOT, 'wikidump_urls.txt')
    if os.path.exists(file_path) and reload == False:
        with open(file_path, 'r') as fp:
            return json.load(fp)    
    
    root = "https://archive.org/download/"
    url1 = 'wiki-20181120'
    url2 = 'wiki-20181101'

    lang2url = {
        "ru": "https://archive.org/download/ruwiki-20181001",
        "el": "https://archive.org/download/elwiki-20181001",
        "uk": "https://archive.org/download/ukwiki-20181001",
        "la": "https://archive.org/download/lawiki-20181001",
    }
    for lang in tqdm(langs): 
        if lang in lang2url:
            continue
        url = root + lang + url1
        rsp = requests.get(url)
        if rsp.status_code == 404:
            url = root + lang + url2
            rsp = requests.get(url)
        
        if rsp.status_code == 404:
            print(f"not found for {lang}")
        else:
            lang2url[lang] = url

    with open(file_path, 'w') as fp:
        json.dump(lang2url, fp, indent=2)
    return lang2url

def get_title_count_per_lang():
    cnt_path = os.path.join(WIKI_URI_ROOT, 'title_cnts.txt')
    lang2cnt = {}
    with open(cnt_path, 'r') as fp:
        for line in fp:
            if line.strip():
                lang, title_cnt = line.strip().split()
                lang2cnt[lang] = int(title_cnt) - 1
    return lang2cnt

def get_article_size_per_lang():
    cnt_path = os.path.join(WIKI_URI_ROOT, 'wiki_article_sizes.txt')
    lang2cnt = {}
    with open(cnt_path, 'r') as fp:
        for line in fp:
            if line.strip():
                lang, size = line.strip().split()
                lang2cnt[lang] = int(size) - 1
    return lang2cnt

def get_abstract_size_per_lang():
    cnt_path = os.path.join(WIKI_URI_ROOT, 'wiki_abstract_sizes.txt')
    lang2cnt = {}
    with open(cnt_path, 'r') as fp:
        for line in fp:
            if line.strip():
                size, lang = line.strip().split()
                lang2cnt[lang] = int(size) - 1
    return lang2cnt

def get_article_gz_size_per_lang():
    cnt_path = os.path.join(WIKI_URI_ROOT, 'wiki_article_gz_sizes.txt')
    lang2cnt = {}
    with open(cnt_path, 'rb') as fp:
        lang2size = json.load(fp)
    return lang2size

def get_abstract_gz_size_per_lang():
    cnt_path = os.path.join(WIKI_URI_ROOT, 'wiki_abstract_gz_sizes.txt')
    lang2cnt = {}
    with open(cnt_path, 'rb') as fp:
        lang2size = json.load(fp)
    return lang2size

def get_title_gz_urls(reload=False):
    file_path = os.path.join(WIKI_URI_ROOT, 'wiki_title_gz_urls.txt')
    if os.path.exists(file_path) and reload == False:
        with open(file_path, 'r') as fp:
            return json.load(fp)    
    
    lang2url = locate_urls()
    lang2title_url = {}
    for lang in langs:
        lang2title_url[lang] = []

    for lang in tqdm(langs):
        html = requests.get(lang2url[lang]).text
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and 'all-titles-in-ns0.gz' in path:
                path = os.path.join(lang2url[lang], path)
                lang2title_url[lang].append(path)
        if len(lang2title_url[lang]) == 1:
            lang2title_url[lang] = lang2title_url[lang][0]
        else:
            raise ValueError(f"There should only be one url of title gz, but get {len(lang2title_url[lang])} in language {lang}")
    
    with open(file_path, 'w') as fp:
        json.dump(lang2title_url, fp, indent=2)
    
    return lang2title_url
    

def get_abstract_gz_urls(reload=False):
    file_path = os.path.join(WIKI_URI_ROOT, 'wiki_abstract_gz_urls.txt')
    if os.path.exists(file_path) and reload == False:
        with open(file_path, 'r') as fp:
            return json.load(fp)    
    
    lang2url = locate_urls()
    lang2abstracts = {}
    for lang in langs:
        lang2abstracts[lang] = []

    for lang in tqdm(langs):
        prefix = lang2url[lang].split('/')[-1]
        pattern = f"{prefix}-abstract[0-9]*.xml.gz"
        
        html = requests.get(lang2url[lang]).text
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and "abstract" in path and re.match(pattern, path):
                path = os.path.join(lang2url[lang], path)
                lang2abstracts[lang].append(path) 
    
        if len(lang2abstracts[lang]) == 0:
            raise ValueError(f"There are no abstract url in language {lang}")
    
    with open(file_path, 'w') as fp:
        json.dump(lang2abstracts, fp, indent=2)
    
    return lang2abstracts

def get_articles_gz_urls(reload=False):
    file_path = os.path.join(WIKI_URI_ROOT, 'wiki_article_gz_urls.txt')
    if os.path.exists(file_path) and reload == False:
        with open(file_path, 'r') as fp:
            return json.load(fp)    
    
    lang2url = locate_urls()
    lang2articles = {}
    for lang in langs:
        lang2articles[lang] = []

    for lang in tqdm(langs, desc="Locating the url for articles"):
        prefix = lang2url[lang].split('/')[-1]
        pattern1 = f"{prefix}-pages-articles[0-9]*.xml-.*.bz2"
        pattern2 = f"{prefix}-pages-articles.xml.bz2"
        
        html = requests.get(lang2url[lang]).text
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and "articles" in path and re.match(pattern1, path):
                path = os.path.join(lang2url[lang], path)
                lang2articles[lang].append(path) 
            
        if len(lang2articles[lang]) == 0:
            for link in soup.find_all('a'):
                path = link.get('href')
                if path and "articles" in path and re.match(pattern2, path):
                    path = os.path.join(lang2url[lang], path)
                    lang2articles[lang].append(path) 
                
        if len(lang2articles[lang]) == 0:        
            raise ValueError(f"There are no article url in language {lang}")
    
    with open(file_path, 'w') as fp:
        json.dump(lang2articles, fp, indent=2)
    
    return lang2articles

def get_gz_size(type="abstract", reload=False):
    if type == "abstract":
        file_path = os.path.join("./", 'wiki_abstract_gz_sizes.txt')
        suffix = "abstract.xml.gz"
    elif type == "article":
        file_path = os.path.join("./", 'wiki_article_gz_sizes.txt')
        suffix = "pages-articles.xml.bz2"
    if os.path.exists(file_path) and reload == False:
        with open(file_path, 'r') as fp:
            return json.load(fp)    
    
    lang2url = locate_urls()
    lang2article_size = {}
    for lang in langs:
        lang2article_size[lang] = []

    for lang in tqdm(langs):
        html = requests.get(lang2url[lang]).text
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find(lambda tag: tag.name == 'a' and tag.get('href', '').endswith(suffix))
        size_tag = a_tag.parent.find_next_sibling('td').find_next_sibling('td') if a_tag else None
        file_size = size_tag.text if size_tag else 'Size not found'
        if file_size.endswith("M"):
            size = float(file_size[:-1]) * 1024
        elif file_size.endswith("G"):
            size = float(file_size[:-1]) * 1024 * 1024
        lang2article_size[lang] = size
    with open(file_path, 'w') as fp:
        json.dump(lang2article_size, fp, indent=2)
    return lang2article_size

"""Download wiki data
"""
SRC_DATA_ROOT = "/disk/xzhao/datasets/wikipedia_2018_octnov"
TGT_DATA_ROOT = "/disk/xzhao/probing-multilingual/2018_dump_wiki_cache"


def ungzip(file_path):    
    with gzip.open(file_path, 'rb') as f_in:
        with open(file_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out) # type: ignore

def download_titles():
    lang2title = get_title_gz_urls()
    for lang in tqdm(langs, desc="Download wiki title compressed files for all languages"):
        lang_path = os.path.join(SRC_DATA_ROOT, "title", lang)
        os.makedirs(lang_path, exist_ok=True)
        remote_url = lang2title[lang]
        gzip_file = os.path.join(lang_path, remote_url.split('/')[-1])
        if not os.path.exists(gzip_file) and not os.path.exists(gzip_file[:-3]):
            request.urlretrieve(remote_url, gzip_file)
            ungzip(gzip_file)
        elif os.path.exists(gzip_file) and not os.path.exists(gzip_file[:-3]):
            ungzip(gzip_file)
        
        if os.path.exists(gzip_file):
            os.remove(gzip_file)

def download_abstracts():
    lang2articles = get_abstract_gz_urls()
    def _download_per_lang(lang):
        lang_path = os.path.join(SRC_DATA_ROOT, "abstract", lang)
        os.makedirs(lang_path, exist_ok=True)
        for remote_url in tqdm(lang2articles[lang], desc=f"Downloading abstract files for {lang}"):
            gzip_file = os.path.join(lang_path, remote_url.split('/')[-1])
            if not os.path.exists(gzip_file) and not os.path.exists(gzip_file[:-3]):
                request.urlretrieve(remote_url, gzip_file)
                ungzip(gzip_file)
            elif os.path.exists(gzip_file) and not os.path.exists(gzip_file[:-3]):
                ungzip(gzip_file)
            
            if os.path.exists(gzip_file):
                os.remove(gzip_file)
                
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_download_per_lang, lang) for lang in langs]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Download wiki abstract compressed files for all languages"):
        future.result()

import progressbar

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_per_file_for_article(remote_url, gzip_file, reload):
    # print(f"Downloading {remote_url} to {gzip_file}")
    if not os.path.exists(gzip_file) or reload == True:
        print(f"Downloading {remote_url} to {gzip_file}")
        request.urlretrieve(remote_url, gzip_file, show_progress)

def download_articles(reload=False):
    lang2abstracts = get_articles_gz_urls()
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for lang in langs:
            lang_path = os.path.join(SRC_DATA_ROOT, "article", lang)
            os.makedirs(lang_path, exist_ok=True)
            for remote_url in tqdm(lang2abstracts[lang], desc=f"Downloading abstract files for {lang}"):
                gzip_file = os.path.join(lang_path, remote_url.split('/')[-1])
                futures.append(executor.submit(download_per_file_for_article, remote_url, gzip_file, reload))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Download wiki articles compressed files for all languages"):
        future.result()

"""Feed wiki data to elastic search
"""
def tokenize_all_subject_object():
    dataset = MaskedDataset(model_name="mbert")
    WIKI_DUMP_RESULT_ROOT = "/home/xzhao/workspace/probing-mulitlingual/datasets/2018_dump_wiki_cache/subject_object"
    os.makedirs(WIKI_DUMP_RESULT_ROOT, exist_ok=True)
    sub_info = dataset.get_sub_info()
    obj_info = dataset.get_obj_info()

    for lang in tqdm(dataset.langs, desc="Writing tokenized subject and object to text file"):
        file_root = os.path.join(WIKI_DUMP_RESULT_ROOT, lang)
        os.makedirs(file_root, exist_ok=True)

        sub_fn = os.path.join(file_root, "subject.text")
        obj_fn = os.path.join(file_root, "object.text")

        with open(sub_fn, 'w') as fp:
            for lang2sub in sub_info.values():
                if lang in lang2sub:
                    fp.write(f"{' '.join(lang2sub[lang]['sub_tokens'])}\n")
        with open(obj_fn, 'w') as fp:
            for lang2obj in obj_info.values():
                if lang in lang2obj:
                    fp.write(f"{' '.join(lang2obj[lang]['obj_tokens'])}\n")

def dump_all_subject_object():
    dataset = MaskedDataset(model_name="mbert")
    WIKI_DUMP_RESULT_ROOT = "/disk/xzhao/probing-multilingual/2018_dump_wiki_cache/subject_object"
    os.makedirs(WIKI_DUMP_RESULT_ROOT, exist_ok=True)
    sub_info = dataset.get_sub_info()
    obj_info = dataset.get_obj_info()

    for lang in tqdm(dataset.langs, desc="Writing tokenized subject and object to text file"):
        file_root = os.path.join(WIKI_DUMP_RESULT_ROOT, lang)
        os.makedirs(file_root, exist_ok=True)

        sub_fn = os.path.join(file_root, "raw_subject.text")
        obj_fn = os.path.join(file_root, "raw_object.text")

        with open(sub_fn, 'w') as fp:
            for lang2sub in sub_info.values():
                if lang in lang2sub:
                    fp.write(f"{lang2sub[lang]['sub']}\n")
        with open(obj_fn, 'w') as fp:
            for lang2obj in obj_info.values():
                if lang in lang2obj:
                    fp.write(f"{lang2obj[lang]['obj']}\n")

import logging
import random

from transformers import BertTokenizer

from constants import LOGGING_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(LOGGING_ROOT, "wikidump_tokenization.log"), filemode="w")
logger = logging.getLogger(__name__)

""" Using abstract for checking cooccurrence of subject-object. 
"""
def find_prefix(titles):
    i = 0
    while all(title[i] == titles[0][i] for title in titles):
        i += 1
    return i, titles[0][:i]

def get_prefix_per_lang():
    lang2fn = {}
    for lang in langs:
        lang_path = os.path.join(SRC_DATA_ROOT, "abstract", lang)
        for fn in os.listdir(lang_path):
            if 'xml' in fn:
                lang2fn[lang] = os.path.join(lang_path, fn)
                break

    lang2prefix = {}
    for lang in tqdm(lang2fn, desc="Finding prefix of titles for each language"):
        tree = ET.iterparse(lang2fn[lang], events=('start', 'end'))
        titles = []
        cnt = 0
        for event, element in tree:
            if event == 'end' and element.tag == 'doc':
                titles.append(element.find('title').text)
                element.clear()
                cnt += 1
                if cnt > 100:
                    break
        lang2prefix[lang] = find_prefix(titles)[0]
    return lang2prefix

# lang2pfxlen = get_prefix_per_lang()
# deprecated
def process_xml_data(xml_data):
    if len(list(xml_data.children)) != 1:
        raise ValueError(f"The number of child element in the XML is not one but {len(list(xml_data.children))}")

    first_child = next(xml_data.children)
    titles = [title.text for title in first_child.find_all('title')]
    prefix_len = find_prefix(random.choices(titles, k=100))[0]
    titles = [title[prefix_len:] for title in titles]
    abstracts = [abstract.text for abstract in first_child.find_all('abstract')]

    if len(titles) != len(abstracts):
        raise ValueError("The number of titles does not match the number of abstracts.")

    return titles, abstracts

def xml_doc_iterator(file_path):
    tree = ET.iterparse(file_path, events=('start', 'end'))
    for event, element in tree:
        if event == 'end' and element.tag == 'doc':
            yield element.find('title').text, element.find('abstract').text

def tokenize_wikidata_and_write(file_path, overwrite=False):
    lang = file_path.split('/')[-2]
    title_tgt_path = os.path.join(TGT_DATA_ROOT, 'titles_tokenized', lang, file_path.split('/')[-1][:-4] + '.txt')
    abs_tgt_path = os.path.join(TGT_DATA_ROOT, 'abstracts_tokenized', lang, file_path.split('/')[-1][:-4] + '.txt')
    
    if os.path.exists(title_tgt_path) and os.path.exists(abs_tgt_path) and overwrite is False:
        logger.info(f"Xml data of {file_path.split('/')[-1]} has already been tokenized")
        return

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    with open(title_tgt_path, 'w') as title_fp, open(abs_tgt_path, 'w') as abstract_fp:
        logger.info(f"Tokenizing title and abstract for file - {file_path}")
        for title, abstract in tqdm(xml_doc_iterator(file_path), desc=f"Tokenizing title and abstract for file - {file_path}"):
            abstract = "" if abstract is None else abstract
            try:
                title_tokens = tokenizer.tokenize(title[lang2pfxlen[lang]:])
                abs_tokens = tokenizer.tokenize(abstract)
                title_fp.write(f"{' '.join(title_tokens)}\n")
                abstract_fp.write(f"{' '.join(abs_tokens)}\n")
            except:
                logger.info(f"Failed to tokenize title {title} with its abstract {abstract}")

def tokenize_all_wikidata(overwrite=False):
    with ProcessPoolExecutor(max_workers=54) as executor:
        for lang in langs:
        # for lang in ['he']:
            logger.info(f"Start to process xml files of language {lang}")
            lang_path = os.path.join(SRC_DATA_ROOT, 'abstract', lang)
            os.makedirs(os.path.join(TGT_DATA_ROOT, 'titles_tokenized', lang), exist_ok=True)
            os.makedirs(os.path.join(TGT_DATA_ROOT, 'abstracts_tokenized', lang), exist_ok=True)
            for fn in sorted(os.listdir(lang_path)):
                if 'xml' in fn:
                    executor.submit(tokenize_wikidata_and_write, os.path.join(lang_path, fn), overwrite)

""" Using title for checking cooccurrence of subject-object. 
"""
def check_entity_existence_in_titles(objects, subjects, lang):
    import numpy as np
    lang_path = os.path.join(SRC_DATA_ROOT, "title", lang)
    assert len(os.listdir(lang_path)) == 1

    obj_prefix_sets = set([obj[:3] for obj in objects])
    sub_prefix_sets = set([sub[:3] for sub in subjects])

    title_fn = os.path.join(lang_path, os.listdir(lang_path)[0])
    titles = set()
    with open(title_fn, 'r') as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if idx == 0 or line == "":
                continue
            try:
                assert len(line.split("\t")) == 2
                title = line.split()[1]
                if title[:3] in obj_prefix_sets or title[:3] in sub_prefix_sets:
                # if title[:3] in sub_prefix_sets:
                    titles.add(title)
            except Exception as e:
                aa = line.split('\t')
                print(f"Assertion error: {aa}")
                
    obj_exist_label = np.zeros((len(objects), ))
    sub_exist_label = np.zeros((len(subjects), ))
    for idx, sub in enumerate(subjects):
        if sub in titles:
            sub_exist_label[idx] = 1

    for idx, obj in enumerate(obj_prefix_sets):
        if obj in titles:
            obj_exist_label[idx] = 1

    return lang, obj_exist_label, sub_exist_label


""" Using article for checking cooccurrence of subject-object. 
"""
import subprocess

from utils import chunk_list, split_list

ARTICLE_ROOT = "/disk/xzhao/datasets/wikipedia_2018_octnov/article"

def run_grep_match_for_en(entities: list[(str, str)], lang, out_fn, line_field=0, fn_name=None, entity_type='sub'):
    out_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", entity_type, lang)
    os.makedirs(out_root, exist_ok=True)
    out_fn = os.path.join(out_root, out_fn)
    if os.path.exists(out_fn[:-3] + 'finished.txt'):
        return "finished"
    
    en_root = os.path.join(ARTICLE_ROOT, lang)
    lang_paths = [os.path.join(en_root, fn) for fn in os.listdir(en_root)]
    lang_paths_ls = chunk_list(lang_paths, 8)
    
    if line_field == 0 and fn_name == None:
        raise ValueError(f"{lang}, {line_field} and {fn_name} get wrong")
    if line_field == 1 and fn_name != None:
        raise ValueError(f"{lang}, {line_field} and {fn_name} get wrong")
    
    with open(out_fn, 'w') as fp:
        for lang_paths in lang_paths_ls:
            for entity, entity_uri in entities:
                p = subprocess.run(f'grep -n "{entity}" {" ".join(lang_paths)}', shell=True, capture_output=True, text=True)
                lines = p.stdout.split('\n')
                file2lines = {}
                for line in lines:
                    if line.strip() == "":
                        continue
                    try:
                        if line_field == 0:
                            line_num = int(line.split(':')[0])
                        else:
                            fn_name, line_num, *_ = line.split(':')
                            line_num = int(line_num)
                        file2lines.setdefault(fn_name, []).append(str(line_num))
                    except:
                        logger.error(f"Failed to get line number and file name from {line}")
                for fn_name, line_nums in file2lines.items():
                    line_nums_str = "__".join(line_nums)
                    newline = f"{entity_uri}@_@{entity}@_@{fn_name}@_@{line_nums_str}"
                    fp.write(f"{newline}\n")
    return out_fn

def run_grep_match(entities: list[(str, str)], lang, out_fn, line_field=0, fn_name=None, entity_type='sub'):
    out_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", entity_type, lang)
    os.makedirs(out_root, exist_ok=True)
    out_fn = os.path.join(out_root, out_fn)
    if os.path.exists(out_fn[:-3] + 'finished.txt'):
        return "finished"
    
    lang_path = os.path.join(ARTICLE_ROOT, lang, '*')
    
    if line_field == 0 and fn_name == None:
        raise ValueError(f"{lang}, {line_field} and {fn_name} get wrong")
    if line_field == 1 and fn_name != None:
        raise ValueError(f"{lang}, {line_field} and {fn_name} get wrong")
    
    with open(out_fn, 'w') as fp:
        for entity, entity_uri in entities:
            p = subprocess.run(f'grep -n "{entity}" {lang_path}', shell=True, capture_output=True, text=True)
            lines = p.stdout.split('\n')
            file2lines = {}
            for line in lines:
                if line.strip() == "":
                    continue
                try:
                    if line_field == 0:
                        line_num = int(line.split(':')[0])
                    else:
                        fn_name, line_num, *_ = line.split(':')
                        line_num = int(line_num)
                    file2lines.setdefault(fn_name, []).append(str(line_num))
                except:
                    logger.error(f"Failed to get line number and file name from {line}")
            for fn_name, line_nums in file2lines.items():
                line_nums_str = "__".join(line_nums)
                newline = f"{entity_uri}@_@{entity}@_@{fn_name}@_@{line_nums_str}"
                fp.write(f"{newline}\n")
    return out_fn

def run_grep_match_in_parallel(entity_type, candidate_langs, thread_num=50):
    dataset = MaskedDataset(model_name="mbert")
    
    if entity_type == 'sub':
        ent_info = dataset.sub_info
        chunk_size = 100 # CANNOT CHANGE, as it related to the file reloading
    elif entity_type == 'obj':
        ent_info = dataset.obj_info
        chunk_size = 25 # CANNOT CHANGE, as it related to the file reloading
    else:
        raise ValueError(f"Unsuppoerted entity type {entity_type}")
    
    lang2arcticle_fns = {}
    final_langs = []
    for lang in candidate_langs:
        if lang in os.listdir(ARTICLE_ROOT):
            lang2arcticle_fns[lang] = os.listdir(os.path.join(ARTICLE_ROOT, lang))
            final_langs.append(lang)

    try:
        with ProcessPoolExecutor(max_workers=thread_num) as executor:
            for lang_idx, lang in enumerate(final_langs):
                futures = []
                if len(lang2arcticle_fns[lang]) == 1:
                    line_field = 0
                    fn_name = lang2arcticle_fns[lang][0]
                else:
                    line_field = 1
                    fn_name = None

                if lang == 'en': # chunk_size = 10 # for english only
                    chunk_size = 10
                entities = [(ent_info[ent_uri][lang][entity_type], ent_uri) for ent_uri in ent_info.keys() if lang in ent_info[ent_uri]]
                sub_splitted = chunk_list(entities, chunk_size)
                for idx, sub_list in enumerate(sub_splitted):
                    futures.append(executor.submit(run_grep_match_for_en, sub_list, lang, f"{lang}-{idx}.txt", line_field=line_field, fn_name=fn_name, entity_type=entity_type))
                    # if lang == 'en':
                    #     futures.append(executor.submit(run_grep_match_for_en, sub_list, lang, f"{lang}-{idx}.txt", line_field=line_field, fn_name=fn_name, entity_type=entity_type))
                    # else:
                    #     futures.append(executor.submit(run_grep_match, sub_list, lang, f"{lang}-{idx}.txt", line_field=line_field, fn_name=fn_name, entity_type=entity_type))
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Matching subject in wiki article by grep for language {lang}, progress: {lang_idx+1}/{len(final_langs)}"):
                    fn = future.result()
                    if fn != "finished":
                        os.rename(fn, fn[:-3] + 'finished.txt')
                logger.info(f"\nfinished grep processing for language {lang}")
    except Exception as e:
        print(f"Processing exception {e}. ")
        logger.info(f"\nEncounter error: {e}")    

import pickle
from collections import defaultdict


def defaultdict2set():
    return defaultdict(set)
    
def get_uri2file2lineids_from_article_grep_matching(root, ent_type, reload=False):
    dump_file = os.path.join(root, 'uri2file2ids.pkl')
    if os.path.exists(dump_file) and reload == False:
        with open(dump_file, 'rb') as fp:
            return pickle.load(fp)
    uri2file2ids = defaultdict(defaultdict2set)
    
    for fn in tqdm(os.listdir(root), desc=f"Reading {ent_type} to line ids in {root}"):
        if 'pkl' in fn:
            continue
        sub_matching_fn = os.path.join(root, fn)
        with open(sub_matching_fn, 'r') as fp:
            for line in fp:
                if line.strip() == "":
                    continue
                uri, _, article_fn, ids = line.strip().split("@_@")
                article_fn = article_fn.split('/')[-1]
                uri2file2ids[uri][article_fn] = set([int(_id) for _id in ids.split('__')])
    with open(dump_file, 'wb') as fp:
        pickle.dump(uri2file2ids, fp)
        
    return uri2file2ids

def _preprocess_uri2file2lineids_from_article_grep_matching(langs):
    futures = []
    with ProcessPoolExecutor(max_workers=50) as exectuor:
        for lang in langs:
            sub_grep_matching_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'sub', lang)    
            obj_grep_matching_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'obj', lang)
            futures.append(exectuor.submit(get_uri2file2lineids_from_article_grep_matching, sub_grep_matching_root, 'subject', True))
            futures.append(exectuor.submit(get_uri2file2lineids_from_article_grep_matching, obj_grep_matching_root, 'object', True))
        for ft in as_completed(futures):
            ft.result()

if __name__ == "__main__":
    # tokenize_all_wikidata(overwrite=True)
    
    # get_articles_gz_urls(reload=True)
    # download_articles(True)
    
    # get_title_gz_urls(reload=True)
    # download_titles()
    
    dump_all_subject_object()
    # candidate_langs = []
    # # removed_langs = ['en']
    # # candidate_langs = ['en']
    # removed_langs = ['en', 'af', 'az', 'ceb', 'da', 'de', 'es', 'hu', 'id', 'it', 'ja', 'nl', 'pl', 'ro', 'ru', 'sk', 'sr', 'sv', 'th', 'vi', 'zh']
    # for lang in langs:
    #     if lang not in removed_langs:
    #         candidate_langs.append(lang)
    # run_grep_match_in_parallel(entity_type='sub', candidate_langs=candidate_langs, thread_num=54)

    # On Tokyo103
    # langs.remove("en")
    # run_grep_match_in_parallel(entity_type='sub', candidate_langs=langs, thread_num=54)

    # _preprocess_uri2file2lineids_from_article_grep_matching(langs)
    # On Tokyo102
    # run_grep_match_in_parallel(entity_type='obj', candidate_langs=['en'], thread_num=54)

    # _preprocess_uri2file2lineids()
    # sub_grep_matching_root = os.path.join(TGT_DATA_ROOT, "article_grep_matched", 'sub', 'ja')
    # get_uri2file2lineids(sub_grep_matching_root, 'subject', reload=True)
    # get_matching_lineids(['ja', 'zh'])