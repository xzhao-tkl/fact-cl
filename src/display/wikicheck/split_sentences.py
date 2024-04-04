from wtpsplit import WtP
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--group')
parser.add_argument('--gpus')

args = parser.parse_args()

INPUT_ROOT_NORMAL = "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article"
INPUT_ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article"

OUTPUT_ROOT_NORMAL = "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article"
OUTPUT_ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article"

if int(args.group) == 5:
    INPUT_ROOT = INPUT_ROOT_106
    OUTPUT_ROOT = OUTPUT_ROOT_106
else:
    INPUT_ROOT = INPUT_ROOT_NORMAL
    OUTPUT_ROOT = OUTPUT_ROOT_NORMAL
    
lang_groups = [
    ['en'], #103 - group0
    ['ceb', 'nl', 'fi', 'id', 'ms', 'da', 'hr', 'lt', 'ga'], # 102 - group1
    ['sv', 'sr', 'pl', 'vi', 'cs', 'af', 'hy'], # 104 - group2
    # ['sv', 'sr', 'pl', 'vi', 'cs', 'af', 'hy', 'ur', 'gl', 'lv', 'cy'], # 104 - group2
    ['ru', 'it', 'pt', 'fa', 'he', 'bg', 'be', 'sk', 'bn', 'sl'], # 101 - group3
    ['de', 'ja', 'uk', 'ca', 'hu', 'ro', 'el', 'ta', 'ka', 'az', 'la'], # 105 - group4
    ['fr', 'es', 'zh', 'ar', 'ko', 'tr', 'th', 'eu', 'hi', 'et', 'sq'],  # 106
    ['ur', 'gl', 'lv', 'cy'] # 104 - group6
]

import re
def is_single_word(s):
    # Regular expression pattern to match a single word
    pattern = r'^\w+$'
    
    # Use re.match to check if the entire string matches the pattern
    return bool(re.match(pattern, s))



import os
import multiprocessing
import GPUtil

def get_free_gpu():
    available_gpus = GPUtil.getAvailable(order='random', limit=1, maxMemory=0.1, maxLoad=0.5)
    if available_gpus:
        return available_gpus[0]
    else:
        return None

def find_lang_from_path(file_path):
    blocks = file_path.split("/")
    is_lang = False
    for block in blocks:
        if is_lang:
            return block
        
        if block == "clean_article":
            is_lang = True

def split_sentence(root):
    input_fn = os.path.join(root, "combined_output.txt")
    output_doc_fn = os.path.join(root, "documents.txt")
    output_doc_fp = open(output_doc_fn, 'w')
    with open(input_fn, 'r') as fp:
        doc = []
        aa = "/"
        for line in tqdm(fp, desc=f"Writing to {root.split(aa)[-1]}"):
            line = line.strip()
            if line == "" or line == "</doc>":
                continue
            if line.startswith("<doc"):
                split_sents = []
                for org_sent in doc:
                    split_sents.append(org_sent)
                output_doc_fp.write(" ".join(split_sents))
                output_doc_fp.write("\n")
                doc = []
            else:
                doc.append(line)
    return output_doc_fn

def process_files_in_parallel(lang_idx):
    path_list = []
    for lang in lang_groups[lang_idx]:
        lang_root = os.path.join(OUTPUT_ROOT, lang)        
        for file_dir in os.listdir(lang_root):
            path_list.append(os.path.join(lang_root, file_dir))
            # print(os.path.join(lang_root, file_dir))
            split_sentence(os.path.join(lang_root, file_dir))
    # print(len(path_list))
    # with multiprocessing.Pool(processes=len(path_list)) as pool:
    #     for file_path in path_list:
    #         pool.apply_async(func=split_sentence, args=(file_path))
    #     pool.close()
    #     pool.join()


def process_files_in_parallel_old(lang_idx):
    path_list = []
    for lang in lang_groups[lang_idx]:
        lang_root = os.path.join(OUTPUT_ROOT, lang)        
        for file_dir in os.listdir(lang_root):
            path_list.append(os.path.join(lang_root, file_dir))
    print(len(path_list))
    with multiprocessing.Pool(processes=len(path_list)) as pool:
        for file_path in path_list:
            gpu = get_free_gpu()
            if gpu is not None:
                pool.apply_async(func=split_sentence, args=(file_path, gpu,))
            else:
                print("No free GPUs available. Waiting for a GPU to become free...")
                pool.close()
                pool.join()
                return
        pool.close()
        pool.join()

# def split_sentence_by_block(root, gpu_id):
#     model = WtP("wtp-bert-mini")
#     model.half().to(f"cuda:{gpu_id}")
    
#     input_fn = os.path.join(root, "combined_output.txt")
#     output_fn = os.path.join(root, "splited_sentences.txt")
#     output_doc_fn = os.path.join(root, "documents.txt")
#     lang = find_lang_from_path(input_fn)

#     print(lang, f"cuda:{gpu_id}", input_fn)

#     output_fp = open(output_fn, 'w')
#     output_doc_fp = open(output_doc_fn, 'w')
#     with open(input_fn, 'r') as fp:
#         doc = []
#         aa = "/"
#         for line in tqdm(fp, desc=f"Runing on {gpu_id}, writing to {root.split(aa)[-1]}"):
#             line = line.strip()
#             if line == "" or line == "</doc>":
#                 continue
#             if line.startswith("<doc"):
#                 split_sents = []
#                 for org_sent in doc:
#                     for split_sent in model.split(org_sent, lang_code=lang):
#                         split_sents.append(split_sent)
#                         output_fp.write(f"{split_sent}\n")
#                 output_doc_fp.write(" ".join(split_sents))
#                 output_fp.write("\n")
#                 output_doc_fp.write("\n")
#                 doc = []
#             else:
#                 doc.append(line)
#     output_fp.close()
#     output_doc_fp.close()
#     return output_doc_fn

# import time

# def read_2K_blocks(file_name, gpu_id):
#     with open(file_name, 'r') as fp:
#         doc = []
#         aa = "/"
#         for line in tqdm(fp, desc=f"Runing on {gpu_id}, processing file to {file_name}"):
#             line = line.strip()
#             if line == "" or line == "</doc>":
#                 continue
#             if line.startswith("<doc"):
#                 split_sents = []
#                 for org_sent in doc:
#                     for split_sent in model.split(org_sent, lang_code=lang):
#                         split_sents.append(split_sent)
#                 output_doc_fp.write(" ".join(split_sents))
#                 output_fp.write("\n")
#                 output_doc_fp.write("\n")
#                 doc = []
#             else:
#                 doc.append(line)

# def process_files_in_parallel_by_block(lang_idx):
#     path_list = []
            
#     with multiprocessing.Pool(processes=int(args.works)) as pool:
#         for lang in lang_groups[lang_idx]:
#             lang_root = os.path.join(OUTPUT_ROOT, lang)        
#             for file_dir in os.listdir(lang_root):
#                 file_dir = os.path.join(lang_root, file_dir)
#                 input_file_name = os.path.join(file_dir, "combined_output.txt")
#                 output_folder = os.path.join(file_dir, "output")
#                 os.makedirs(output_folder, exist_ok=True)

#                 for file_path in path_list:
#                     gpu = get_free_gpu()
#                     while gpu is None:
#                         gpu = get_free_gpu()
#                         time.sleep(5)
#                     pool.apply_async(func=split_sentence, args=(file_path, gpu,))
#         pool.close()
#         pool.join()
process_files_in_parallel(int(args.group))
