import argparse
import os
import shutil
import subprocess

from tqdm import tqdm

ARTICLE_ROOT = "/disk/xzhao/datasets/wikipedia_2018_octnov/article"
CLEAN_ROOT = "/disk/xzhao/datasets/wikipedia_2018_octnov/clean_article"

ARTICLE_ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/article"
CLEAN_ROOT_106 = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/wikipedia_2018_octnov/clean_article"

lang_groups = [
    ['en'],
    ['ceb', 'nl', 'fi', 'id', 'ms', 'da', 'hr', 'lt', 'ga'],
    ['sv', 'sr', 'pl', 'vi', 'cs', 'af', 'hy', 'ur', 'gl', 'lv', 'cy'],
    ['ru', 'it', 'pt', 'fa', 'he', 'bg', 'be', 'sk', 'bn', 'sl'],
    ['de', 'ja', 'uk', 'ca', 'hu', 'ro', 'el', 'ta', 'ka', 'az', 'la'],
    ['fr', 'es', 'zh', 'ar', 'ko', 'tr', 'th', 'eu', 'hi', 'et', 'sq']
]

parser = argparse.ArgumentParser()
parser.add_argument('--group')
parser.add_argument('--func')

args = parser.parse_args()

if int(args.group) == 5:
    INPUT_ROOT = ARTICLE_ROOT_106
    OUTPUT_ROOT = CLEAN_ROOT_106
else:
    INPUT_ROOT = ARTICLE_ROOT
    OUTPUT_ROOT = CLEAN_ROOT

print(OUTPUT_ROOT)
def extract(input_fn, output_dir):
    if os.path.exists(output_dir):
        return
    command = ['python', '-m', 'wikiextractor.WikiExtractor', input_fn, '-o', output_dir]
    # Add the optional template file argument if provided.
    try:
        subprocess.run(command, check=True)
        print("WikiExtractor completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running WikiExtractor: {e}")

def extract_langs(lang_group_idx):
    for lang in lang_groups[lang_group_idx]:
        lang_root = os.path.join(INPUT_ROOT, lang)
        output_root = os.path.join(OUTPUT_ROOT, lang)
        os.makedirs(output_root, exist_ok=True)
        for file_name in tqdm(os.listdir(lang_root)):
            input_fn = os.path.join(lang_root, file_name)
            output_dir = os.path.join(output_root, file_name)
            extract(input_fn, output_dir)

def combine_dir(directory_path):
    output_file_name = os.path.join(directory_path, 'combined_output.txt')
    if os.path.exists(output_file_name):
        return
    
    # Open the output file in write mode.
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        # Loop through all files in the directory.
        for sub_dir in os.listdir(directory_path):
            sub_dir = os.path.join(directory_path,sub_dir)
            if not os.path.isdir(sub_dir):
                continue

            for filename in os.listdir(sub_dir):
                file_path = os.path.join(sub_dir, filename)
                # Check if the item is a file (not a subdirectory).
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as input_file:
                        # Read the content of the input file and write it to the output file.
                        output_file.write(input_file.read())
        for sub_dir in os.listdir(directory_path):
            sub_dir = os.path.join(directory_path,sub_dir)
            if os.path.isdir(sub_dir):
                shutil.rmtree(sub_dir)
    print(f'Combined files into: {output_file_name}')

def combine_all(lang_group_idx):
    for lang in lang_groups[lang_group_idx]:
        lang_root = os.path.join(OUTPUT_ROOT, lang)
        print(lang_root)
        for file_dir in os.listdir(lang_root):
            file_dir = os.path.join(lang_root, file_dir)
            if os.path.isdir(file_dir):
                combine_dir(file_dir)
            


if args.func == "extract":
    extract_langs(int(args.group))
elif args.func == "combine":
    combine_all(int(args.group))
else:
    raise Exception()
