from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from transformers import BertTokenizer
import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--group')
parser.add_argument('--cpus')

args = parser.parse_args()
cpus = int(args.cpus)

ROOT_NORMAL = "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article"
ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article"

if int(args.group) == 5:
    ROOT = ROOT_106
else:
    ROOT = ROOT_NORMAL
    
lang_groups = [
    ['en'],
    ['ceb', 'nl', 'fi', 'id', 'ms', 'da', 'hr', 'lt', 'ga'],
    ['sv', 'sr', 'pl', 'vi', 'cs', 'af', 'hy', 'ur', 'gl', 'lv', 'cy'],
    ['ru', 'it', 'pt', 'fa', 'he', 'bg', 'be', 'sk', 'bn', 'sl'],
    ['de', 'ja', 'uk', 'ca', 'hu', 'ro', 'el', 'ta', 'ka', 'az', 'la'],
    ['fr', 'es', 'ar', 'ko', 'tr', 'th', 'eu', 'hi', 'et', 'sq'],
    ['ur', 'gl', 'lv', 'cy'] # 104 - group6
]

def chunk_list(input_list, chunk_size):
    res = []
    for i in range(0, len(input_list), chunk_size):
        res.append(input_list[i:i+chunk_size])
    return res


def tokenize_docs(file_name):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    root = os.path.dirname(file_name)
    suffix = file_name.split('.')[-1]
    output_name = os.path.join(root, f"tokenized_doc.txt.{suffix}")
    output_fp = open(output_name, 'w')
    with open(file_name, 'r') as fp:
        for line in tqdm(fp, desc=file_name.split('/')[-2]):
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            for chunk in chunk_list(tokens, 512):
                output_fp.write(" ".join(chunk) + "\n")
            output_fp.write("\n")
    output_fp.close()


def tokenize_all(lang_idx):
    path_list = []
    for lang in lang_groups[lang_idx]:
        lang_root = os.path.join(ROOT, lang)
        for file_dir in os.listdir(lang_root):
            file_dir = os.path.join(lang_root, file_dir)
            for file_name in os.listdir(file_dir):
                if file_name.startswith("documents.txt.0"):
                    file_name = os.path.join(file_dir, file_name)
                    path_list.append(file_name)
        
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        futures = [executor.submit(tokenize_docs, path) for path in path_list]
        for future in as_completed(futures):
            future.result()

tokenize_all(int(args.group))
# tokenize_docs("/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article/es/eswiki-20181120-pages-articles2.xml-p143638p597334/documents.txt.01")