from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from transformers import BertTokenizer
import argparse
import os
from tqdm import tqdm
import subprocess

lang_groups = [
    ['en'], #103 - group0
    ['ceb', 'nl', 'fi', 'id', 'ms', 'da', 'hr', 'lt', 'ga'], # 102 - group1
    ['sv', 'sr', 'pl', 'vi', 'cs', 'af', 'hy', 'ur', 'gl', 'lv', 'cy'], # 104 - group2
    ['ru', 'it', 'pt', 'fa', 'he', 'bg', 'be', 'sk', 'bn', 'sl'], # 101 - group3
    ['de', 'ja', 'uk', 'ca', 'hu', 'ro', 'el', 'ta', 'ka', 'az', 'la'], # 105 - group4
    ['fr', 'es', 'zh', 'ar', 'ko', 'tr', 'th', 'eu', 'hi', 'et', 'sq'],  # 106
    ['ur', 'gl', 'lv', 'cy'] # 104 - group6
]
parser = argparse.ArgumentParser()
parser.add_argument('--group')
parser.add_argument('--cpus')

args = parser.parse_args()
cpus = int(args.cpus)

ROOT_NORMAL = "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article"
ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article"

OUTPUT_ROOT = "/home/xzhao/workspace/probing-mulitlingual/datasets/2018_dump_wiki_cache/sub_obj_ggrep"
GGREP_PATH = "/home/xzhao/workspace/probing-mulitlingual/src/scripts/ggrep"
ENTITIT_ROOT = "/home/xzhao/workspace/probing-mulitlingual/result/dataset-mbert/subject_object"


if int(args.group) == 5:
    ROOT = ROOT_106
else:
    ROOT = ROOT_NORMAL


def ggrep(entity_fn, tokenized_article_fn, out_fn):
    command = f"./ggrep {entity_fn}"
    with open(tokenized_article_fn, 'r') as infile, open(out_fn, 'w') as outfile:
        subprocess.run(command, stdin=infile, stdout=outfile, shell=True)

def ggrep_all(lang_idx):
    path_list = []
    for lang in lang_groups[lang_idx]:
        lang_root = os.path.join(ROOT, lang)
        for file_dir in os.listdir(lang_root):
            file_dir = os.path.join(lang_root, file_dir)
            for file_name in os.listdir(file_dir):
                if file_name.startswith("tokenized_doc.txt.0"):
                    path_list.append((lang, os.path.join(file_dir, file_name)))

    with ProcessPoolExecutor(max_workers=cpus) as executor:
        futures = []
        for lang, tokenized_article_fn in path_list:
            sub_folder = tokenized_article_fn.split('/')[-2]
            os.makedirs(os.path.join(OUTPUT_ROOT, lang, sub_folder), exist_ok=True) 
            sub_match_fn = os.path.join(OUTPUT_ROOT, lang, sub_folder, f"subject_match.txt.{tokenized_article_fn.split('.')[-1]}")
            obj_match_fn = os.path.join(OUTPUT_ROOT, lang, sub_folder, f"object_match.txt.{tokenized_article_fn.split('.')[-1]}")
            sub_token_fn = os.path.join(ENTITIT_ROOT, lang, "subject.text")
            obj_token_fn = os.path.join(ENTITIT_ROOT, lang, "object.text")
            # print(sub_token_fn, tokenized_article_fn, sub_match_fn)
            futures.append(executor.submit(ggrep, sub_token_fn, tokenized_article_fn, sub_match_fn))
            futures.append(executor.submit(ggrep, obj_token_fn, tokenized_article_fn, obj_match_fn))
            
        for future in tqdm(as_completed(futures)):
            future.result()


# ggrep("/home/xzhao/workspace/probing-mulitlingual/result/dataset-mbert/subject_object/en/subject.text",
#       "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article/en/enwiki-20181120-pages-articles27.xml-p45663464p47163464/tokenized_doc.txt.00",
#       "/disk/xzhao/datasets/wikipedia_2018_octnov/sub_obj_ggrep/en/subject_match.txt.00")
            
ggrep_all(int(args.group))