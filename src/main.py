import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.pred_evaluation import Evaluator
from mask_dataset import MaskedDataset
from mask_prediction import predict_all_parallel
from modules.bert_base_model import BERTBaseModel
from modules.xlmr_base_model import XLMBaseModel

CACHE_FOLDER = "/home/xzhao/workspace/probing-mulitlingual/src/.cache"


def set_device(args):
    if args.no_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device(f"cuda:{args.gpus}")
        n_gpu = 1
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument(
        "--model_type",
        default="xlmr",
        type=str,
        required=False,
        help="The model used for analysis",
    )
    parser.add_argument(
        "--neuron_type",
        default="hidden",
        type=str,
        required=False,
        choices=["ffn-activation", "ffn-output", "layer-emb"],
        help="The types of neurons used for analaysis, must be one of [ffn-activation, ffn-output, layer-emb]",
    )

    parser.add_argument(
        "--data_type",
        default="mlama",
        type=str,
        required=False,
        help="The data path to read from",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        required=False,
        help="The language for analysis",
    )
    parser.add_argument(
        "--lang2",
        default="zh",
        type=str,
        required=False,
        help="Another language to analyze cross-lingual transferability",
    )

    parser.add_argument(
        "--no_cuda", type=bool, default=True, help="If cuda is available"
    )
    parser.add_argument("--gpus", type=str, default="0", help="available gpus id")

    parser.add_argument(
        "--predict_mask",
        type=bool,
        required=True,
        help="Whether to perform mask prediction",
    )

    parser.add_argument(
        "--evaluate_mask_prediction",
        type=bool,
        required=True,
        help="Whether to evaluate mask prediction",
    )
    # parse arguments
    args = parser.parse_args()

    # set device
    device = set_device(args)

    # load dataset
    dataset = MaskedDataset("mlama")

    # Set MLLM model
    if args.model_type == "xlmr":
        Model = XLMBaseModel
    elif args.model_type == "mbert":
        Model = BERTBaseModel
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented")

    # Mask prediction
    if args.predict_mask:
        predict_all_parallel(dataset, Model, 8)

    # Evaluate predicted mask tokens
    if args.evaluate_mask_prediction:
        evaluator = Evaluator(dataset)
        evaluator.evaluate_p1_score()
        evaluator.evaluate_single_multi_token_p1()
        evaluator.draw_obj_distributions()

    # Collect activations by feeding prompts in dataset to MLLM.
    model = Model(device)
    collect_neurons(dataset, model, args.lang)
