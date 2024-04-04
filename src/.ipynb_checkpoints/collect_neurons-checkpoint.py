import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from mask_dataset import MaskedDataset
from modules.xlmr_base_model import XLMBaseModel
from modules.bert_base_model import BERTBaseModel


CACHE_FOLDER = "/home/xzhao/workspace/probing-mulitlingual/src/.cache"

def set_device(args):
    if args.no_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    return device

def collect_neurons(dataset, model, lang, relid=None, pkl_io=True):
    def _collect_neurons(dataset, model, lang, relid, pkl_io=True):
        dump_path = os.path.join(CACHE_FOLDER, "{}-{}-neurons.pkl".format(lang, relid))
        if os.path.exists(dump_path):
            return
        data = dataset.get_lang_type(lang, relid)
        sentences = data['sent'].tolist()
        logits, acts, outs = model.collect_neurons(sentences)
        mask_ind, pred_tokens = model.collect_topk_pred(sentences, 100, logits)
        assert len(sentences) == len(logits) == len(acts) == len(outs) == len(pred_tokens)

        data.insert(len(data.columns), "acts", acts)
        data.insert(len(data.columns), "outs", outs)
        data.insert(len(data.columns), "maskid", mask_ind)
        data.insert(len(data.columns), "preds", pred_tokens)
        if pkl_io:
            data.to_pickle(dump_path)
        del data
        return logits, acts, outs, mask_ind, pred_tokens    
    print("Start to collect neurons form language {} from dataset {}".format(lang, dataset.name))
    if relid == None or pkl_io is False:
        for relid in tqdm(dataset.get_lang(lang)['relid'].unique()):
            _collect_neurons(dataset, model, lang, relid, pkl_io)
    else:
        _collect_neurons(dataset, model, lang, relid, pkl_io)

def load_neurons(lang, relid):
    dump_path = os.path.join(CACHE_FOLDER, "{}-{}-neurons.pkl".format(lang, relid))
    if os.path.exists(dump_path):
        return pd.read_pickle(dump_path)
    else:
        raise ValueError("{}-{}-neurons.pkl doesn't exist.".format(lang, relid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--model_type",
                        default='xlm-mlm',
                        type=str,
                        required=False,
                        help="The model used for analysis")
    parser.add_argument("--neuron_type",
                        default='hidden',
                        type=str,
                        required=False,
                        choices=['ffn-activation', 'ffn-output', 'layer-emb'],
                        help="The types of neurons used for analaysis, must be one of [ffn-activation, ffn-output, layer-emb]")

    parser.add_argument("--data_type",
                        default='mlama',
                        type=str,
                        required=False,
                        help="The data path to read from")
    parser.add_argument("--lang",
                        default='en',
                        type=str,
                        required=False,
                        help="The language for analysis")
    parser.add_argument("--lang2",
                        default='zh',
                        type=str,
                        required=False,
                        help="Another language to analyze cross-lingual transferability")
    
    parser.add_argument("--no_cuda",
                        type=bool,
                        default=True,
                        help="If cuda is available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")

    parser.add_argument("--predict_mask",
                        type=bool,
                        required=True,
                        help="Whether to perform mask prediction")
    
    parser.add_argument("--evaluate_mask_prediction",
                        type=bool,
                        required=True,
                        help="Whether to evaluate mask prediction")
    # parse arguments
    args = parser.parse_args()

    # set device
    device = set_device(args)
    
    # load dataset            
    mlama = MaskedDataset("mlama")
    
    # Set MLLM model
    if args.model_type == "xlm-mlm":
        Model = XLMBaseModel
    elif args.model_type == "mbert":
        Model = BERTBaseModel
    
    lang2objs = {}
    for lang in mlama.langs:
        objs = load_objects(lang, model.name, model)
        lang2objs.update({lang:objs})
        
    # Mask prediction
    if args.predict_mask:
        predict_all_parallel(mlama, Model, 8)
    
    # Evaluate predicted mask tokens
    if args.evaluate_mask_prediction:
        
        
    # Collect activations by feeding prompts in dataset to MLLM. 
    collect_neurons(mlama, xlmr, args.lang)
    collect_neurons(mlama, xlmr, args.lang2)
    
    