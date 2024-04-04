import os
import ast
import json
import pandas as pd
from tqdm import tqdm
import _pickle as pickle
from utils import load_objects, strip_space

CACHE_PATH = "/home/xzhao/workspace/probing-mulitlingual/datasets/.cache"
DATASET_PATH = {
    "mlama": "/home/xzhao/workspace/probing-mulitlingual/datasets/mlama1.1"
}
TOKENIZED_DATAFRAME_ROOT = "/home/xzhao/workspace/probing-mulitlingual/datasets/mlama_relations"
pd.set_option('mode.chained_assignment', None)

class MaskedDataset:
    def __init__(self, data_type, model_name, reload=False):
        self.datapath = DATASET_PATH[data_type]
        self.langs = []
        self.name = data_type
        self.model_name = model_name
        if self.model_name == "xlmr":
            self.tokenized_obj_root = os.path.join(TOKENIZED_DATAFRAME_ROOT, "xlm-roberta-large")
            self.mask_token = "<mask>"
        elif self.model_name == "mbert":
            self.tokenized_obj_root = os.path.join(TOKENIZED_DATAFRAME_ROOT, "bert-base-multilingual-cased")
            self.mask_token = "[MASK]"
        else:
            raise Exception("Undefined model name {}".format(model_name))
            
        if data_type == "mlama":
            data = []
            for lang in tqdm(os.listdir(self.datapath)):
                data.append(self._read_lama_dataset(self.datapath, lang, reload))
                self.langs.append(lang)
            self.data = pd.concat(data)
        else:
            raise Exception("The type of data is not correctly specified")

    def _read_lama_dataset(self, path, lang, reload=False):
        dumppath = os.path.join(CACHE_PATH, lang + "_mlama.feather")
        if os.path.exists(dumppath) and reload is False:
            return pd.read_feather(dumppath)
        dataset = pd.DataFrame(columns=['sent', 'lang', 'relid', 'lineid', 'obj_uri', 'uuid'])

        path = os.path.join(path, lang)
        template_path = os.path.join(path, "templates.jsonl")
        with open(template_path, 'r') as fp:
            templates = [json.loads(l) for l in fp.readlines()]
        for type in templates:
            rel_id = type["relation"]
            temp = type["template"]
            relpath = os.path.join(path, rel_id + ".jsonl")
            if os.path.exists(relpath) is False:
                continue
            with open(relpath, 'r') as fp:
                for rel in [json.loads(l) for l in fp.readlines()]:
                    item = {
                        'sent': [temp.replace('[X]', rel["sub_label"])],
                        'lang': [lang],
                        'relid': [rel_id],
                        'obj': [rel["obj_label"]],
                        'obj_uri': [rel["obj_uri"]],
                        'uuid': [rel["uuid"]],
                        'lineid': [str(rel['lineid'])]}
                    dataset = pd.concat([dataset, pd.DataFrame(item)])
        dataset.reset_index().to_feather(dumppath)
        return dataset
    
    def replace_with_mask(self, sentences, mask_num):
        return [s.replace('[Y]', " ".join(mask_num*(self.mask_token,))) for s in sentences]

    def get_lang(self, lang):
        return self.data[self.data['lang'] == lang]

    def get_lang_type(self, lang, rel, tokenizer=None):
        dump_file = os.path.join(self.tokenized_obj_root, lang, '{}.csv'.format(rel))
        if os.path.exists(dump_file):
            return pd.read_csv(dump_file, index_col=0)
        elif tokenizer != None:
            return self._save_tokenize_objs(lang, rel, tokenizer)
        return self.data[(self.data['lang'] == lang) & (self.data['relid'] == rel)]
    
    def get_rels_in_lang(self, lang):
        for rel in load_objects(lang, self.model_name).keys():
            yield rel
        
    def get_line(self, relid, lineid):
        return self.data[(self.data['relid'] == relid) & (self.data['lineid'] == lineid)]
    
    def get_obj_uris(self):
        dump_file = os.path.join(CACHE_PATH, "obj_uris.pkl")
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as fp:
                return pickle.load(fp)
        obj_uris = {}
        for lang in tqdm(self.langs):
            for rel in self.get_rels_in_lang(lang):
                gold = self.get_lang_type(lang, rel)
                for idx in gold.index:
                    obj = gold.loc[idx]['obj']
                    obj_uri = gold.loc[idx]['obj_uri']
                    obj_ids = strip_space(ast.literal_eval(gold.loc[idx]['obj_ids']))
                    obj_tokens = strip_space(ast.literal_eval(gold.loc[idx]['obj_tokens']))
                    if obj_uri not in obj_uris:
                        obj_uris.update({obj_uri: {lang: {"obj": obj, "obj_ids": obj_ids, "obj_tokens": obj_tokens}}})
                    elif lang not in [obj_uris[obj_uri]]:
                        obj_uris[obj_uri].update({lang: {"obj": obj, "obj_ids": obj_ids, "obj_tokens": obj_tokens}})
        with open(dump_file, 'wb') as fp:
            return pickle.dump(obj_uris, fp)
        return obj_uris
        
    def _save_tokenize_objs(self, lang, rel, tokenizer):
        if not os.path.exists(self.tokenized_obj_root):
            os.mkdir(self.tokenized_obj_root)
        if not os.path.exists(os.path.join(self.tokenized_obj_root, lang)):
            os.mkdir(os.path.join(self.tokenized_obj_root, lang))
        # print("Save tokenized objects for {}-{}".format(lang, rel))
        df = self.data[(self.data['lang'] == lang) & (self.data['relid'] == rel)]
        df.loc[:, 'obj_tokens'] = df['obj'].apply(lambda x: tokenizer.tokenize(x))
        df.loc[:, 'obj_ids'] = df['obj_tokens'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))
        df.to_csv(os.path.join(self.tokenized_obj_root, lang, '{}.csv'.format(rel)))
        return df

if __name__ == "__main__":
    from mask_prediction import load_objects
    from transformers import AutoTokenizer, BertTokenizer
    
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#     mlama = MaskedDataset("mlama", tokenizer.mask_token, 1)
#     for lang in tqdm(mlama.langs):
#         objs = load_objects(lang, "xlmr", None)
#         for rel in objs.keys():
#             mlama._save_tokenize_objs(lang, rel, tokenizer)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    mlama = MaskedDataset("mlama", "mbert")

    from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
    with ThreadPoolExecutor(60) as executor:
        lang_rel = []
        for lang in mlama.langs:
            objs = load_objects(lang, "mbert", None)
            for rel in objs.keys():
                lang_rel.append([lang, rel])
        futures = [executor.submit(mlama._save_tokenize_objs, lang, rel, tokenizer) for lang, rel in lang_rel]
        for future in as_completed(futures):
            future.result()