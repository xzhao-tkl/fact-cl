import os
import json
import pickle


def parse_list(pred):
    return ast.literal_eval(pred)

def tokens2id(pred, tokenizer):
    token_ids = []
    for token in pred:
        assert(len(token) == 1)
        token_ids.append(tokenizer.convert_tokens_to_ids(token)[0])
    return token_ids

def strip_space(ids):
    new_ids = []
    for i, _id in enumerate(ids):
        if _id != 6:
            # if i!=0 or i!=len(ids)-1:
            #     raise Exception('Space appears in the middle of the token sequence')
            new_ids.append(_id)
    return new_ids

def adding_tokenization_to_prediction(lang, rel, root):
    tgt_fn = os.path.join(root, lang, "{}-{}.csv".format(lang, rel))
    if not os.path.exists(tgt_fn):
        return pd.DataFrame()
    df = pd.read_csv(tgt_fn)
    if 'pred_ids' in df.columns:
        # print("The conversion is already done")
        return
    df['prediction'] = df['prediction'].apply(lambda x: parse_list(x))
    
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    df['pred_ids'] = df['prediction'].apply(lambda x: tokens2id(x, tokenizer))
    if 'Unnamed: 0.1' in df.columns:
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])    
    df.to_csv(tgt_fn)
    print("Finished Converstion {}-{}. Rewrite {}".format(lang, rel, tgt_fn))
    return df

def load_objects(lang, model_name, model=None):
    bin_path = "../datasets/TREx_multilingual_objects/" + "{}-{}".format(model_name, lang) + ".pkl"
    if os.path.exists(bin_path):
        with open(bin_path, 'rb') as fp:
            return pickle.load(fp)
    
    object_path = "../datasets/TREx_multilingual_objects/" + lang + ".json"
    with open(object_path) as f:
        candidates = json.load(f) 
    objs_num_dict = {}
    for rel in candidates.keys():
        objs_num_dict.update({rel:{}})
        maxlen = 0
        for obj in candidates[rel]['objects']:
            # maxlen = max(maxlen, len(obj))
            maxlen = max(maxlen, len(model.tokenizer.tokenize(obj)))
        for i in range(1, maxlen+1):
            objs_num_dict[rel].update({i:{}})
        for obj in candidates[rel]['objects']:
            tokens = model.tokenizer.tokenize(obj)
            obj_len = len(tokens)
            objs_num_dict[rel][obj_len].update({obj:model.tokens_to_ids(tokens)})
    
    with open(bin_path, 'wb') as fp:
        pickle.dump(objs_num_dict, fp)
    
    return objs_num_dict
