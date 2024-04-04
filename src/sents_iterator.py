import ast
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import EVALUATION_ROOT, PREDICTION_ROOT
from pred_evaluation import read_pred
from utils import strip_space
from constants import NEURONS_ROOT


# Generate matches information by join gold and evl dataframes. The generated data can be used for neuron probing
def generate_match_sent_by_full_match(dataset, lang, rel):
    gold = dataset.get_lang_type(lang, rel)
    if gold is None:
        print(f"No matches in {lang}-{rel} for model {dataset.model_name}")
        return pd.DataFrame()

    evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])

    df = pd.DataFrame(
        columns=gold.columns.tolist()
        + ["matched", "mask_num", "match_sent", "match_prediction", "match_pred_ids"],
        index=gold.index.tolist(),
    )
    for idx in gold.index:
        gold_row = gold.loc[idx]
        df.loc[idx] = gold_row
        df.loc[idx].matched = False

        obj_ids = strip_space(ast.literal_eval(gold_row["obj_ids"]))
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = evl[evl["id"] == idx]
        for index, row in preds.iterrows():
            pred_list = strip_space(ast.literal_eval(row.pred_ids))
            if obj_ids == pred_list:
                df.loc[idx].matched = True
                df.loc[idx].mask_num = row.mask_num
                df.loc[idx].match_sent = row.sent
                df.loc[idx].match_prediction = row.prediction
                df.loc[idx].match_pred_ids = row.pred_ids
    return df


def generate_match_sent_by_partial_match(dataset, lang, rel):
    gold = dataset.get_lang_type(lang, rel)
    if gold is None:
        return pd.DataFrame()

    evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])

    df = pd.DataFrame(
        columns=gold.columns.tolist()
        + ["matched", "mask_num", "match_sent", "match_prediction", "match_pred_ids"],
        index=gold.index.tolist(),
    )
    for idx in gold.index:
        gold_row = gold.loc[idx]
        df.loc[idx] = gold_row
        df.loc[idx].matched = False

        obj_ids = strip_space(ast.literal_eval(gold_row["obj_ids"]))
        """
        The matching between golden data and prediction is through comparing the the "id" column. 
        ** This methods is limited to be used in (language, relation)-specific dataframe.**
        Because the "id" property is unique for each prompt and consistent between `gold` and `evl`.
        """
        preds = evl[evl["id"] == idx]

        partial_matched = {
            "mask_num": [],
            "match_sent": [],
            "match_prediction": [],
            "match_pred_ids": [],
        }
        for index, row in preds.iterrows():
            pred_list = strip_space(ast.literal_eval(row.pred_ids))
            if all(x in pred_list for x in obj_ids):
                df.loc[idx].matched = True
                partial_matched["mask_num"].append(row.mask_num)
                partial_matched["match_sent"].append(row.sent)
                partial_matched["match_prediction"].append(row.prediction)
                partial_matched["match_pred_ids"].append(row.pred_ids)
        df.loc[idx].mask_num = partial_matched["mask_num"]
        df.loc[idx].match_sent = partial_matched["match_sent"]
        df.loc[idx].match_prediction = partial_matched["match_prediction"]
        df.loc[idx].match_pred_ids = partial_matched["match_pred_ids"]
    return df


def generate_match_sentences(dataset, reload=False):
    print(
        f"Start to generate sentences matched by full-match for model {dataset.model_name}"
    )
    for lang, rel in dataset.lang_rel_iter():
        root = os.path.join(
            EVALUATION_ROOT[dataset.model_name], "matched_sentences", rel
        )
        os.makedirs(root, exist_ok=True)

        dump_file = os.path.join(
            EVALUATION_ROOT[dataset.model_name],
            "matched_sentences",
            rel,
            f"{lang}-full-match.csv",
        )
        if os.path.exists(dump_file) and reload == False:
            continue
        df = generate_match_sent_by_full_match(dataset, lang, rel)
        df.to_csv(dump_file, index=False)

    print(
        f"Start to generate sentences matched by partial-match for model {dataset.model_name}"
    )
    for lang, rel in dataset.lang_rel_iter():
        root = os.path.join(
            EVALUATION_ROOT[dataset.model_name], "matched_sentences", rel
        )
        os.makedirs(root, exist_ok=True)

        dump_file = os.path.join(
            EVALUATION_ROOT[dataset.model_name],
            "matched_sentences",
            rel,
            f"{lang}-partial-match.csv",
        )
        if os.path.exists(dump_file) and reload == False:
            continue
        gold = dataset.get_lang_type(lang, rel)
        evl = read_pred(lang, rel, PREDICTION_ROOT[dataset.model_name])
        df = generate_match_sent_by_partial_match(dataset, lang, rel)
        df.to_csv(dump_file, index=False)


class SentIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_matches_by_obj(self, obj_uri, correct_matches) -> list:
        """Return matched mask sentences indexed by objects
        Returns:
            matches: [sent1, sent2, ...]
        """
        if obj_uri not in correct_matches:
            return []
        
        matches = []
        for lang2sent in correct_matches[obj_uri].values():
            matches.extend(iter(lang2sent.values()))
        return matches

    def get_matches_by_obj_lang(self, obj_uri, lang, correct_matches):
        """Return matched mask sentences indexed by objects and language
        Returns:
            matches: {'en': [sent1, sent2], 'fr': [sent3, sent4], ...}
        """

        if obj_uri not in correct_matches:
            return []

        return [
            lang2sent[lang]
            for lang2sent in correct_matches[obj_uri].values()
            if lang in lang2sent
        ]

    def get_matches_by_sub(self, rel, sub_uri, correct_matches) -> list:
        """Return matched mask sentences indexed by subjects
        Returns:
            matches: [sent1, sent2, ...]
        """
        matches = set()
        uuid_info = self.dataset.get_uuid_info()[rel]
        target_uuids = [uuid for uuid, info in uuid_info.items() if info['sub_uri'] == sub_uri]
        for uuid2sents in correct_matches.values():
            sents = sum(
                (
                    list(uuid2sents[uuid].values())
                    for uuid in target_uuids
                    if uuid in uuid2sents
                ),
                [],
            )
            for sent in sents:
                matches.add(sent)
        return list(matches)

    def get_matches_by_sub_lang(self, rel, sub_uri, lang, correct_matches):
        """Return matched mask sentences indexed by subjects and language
        Returns:
            matches: [sent1, sent2]
        """
        matches = set()
        uuid_info = self.dataset.get_uuid_info()[rel]
        target_uuids = [uuid for uuid, info in uuid_info.items() if info['sub_uri'] == sub_uri]
        for uuid2sents in correct_matches.values():
            for uuid in target_uuids:
                if uuid in uuid2sents and lang in uuid2sents[uuid]:
                    matches.add(uuid2sents[uuid][lang])
        return list(matches)

    def get_all_matches(self, correct_matches) -> list:
        """Return matched mask sentences indexed by relation
        Returns:
            matches: [sent1, sent2, ...]
        """
        matches = []
        for uuid2lang in correct_matches.values():
            for lang2sent in uuid2lang.values():
                matches.extend(iter(lang2sent.values()))
        return matches

    def get_matches_by_lang(self, lang, correct_matches):
        """Return matched mask sentences indexed by relation and language
        Returns:
            matches: {'en': [sent1, sent2], 'fr': [sent3, sent4], ...}
        """
        matches = []
        for uuid2langs in correct_matches.values():
            matches.extend(
                lang2sent[lang]
                for lang2sent in uuid2langs.values()
                if lang in lang2sent
            )
        return matches

    def get_matches_by_lang_rel(self, lang, rel):
        correct_matches = self.get_correct_matches(rel)
        return self.get_matches_by_lang(lang, correct_matches)
        

    def get_correct_matches(self, rel, match_type="full-match"):
        """
        Returns
            uri2sents: {
                "Q150": { #obj_uri
                    '004bca2b-2e00-4872-a644-06eb29a10f55': { # uuid
                        'ca': 'La llengua nativa de Pierre Blanchar és [MASK] [MASK] [MASK].',
                        'en': 'The native language of Pierre Blanchar is [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] .',
                        ...
                    }, ...
                }, ...
            }
        """
        sents_dfs = {
            lang: self._get_match_sentences(lang, rel, match_type)
            for lang in self.dataset.langs
        }
        uri2sents = {}
        for lang, df in sents_dfs.items():
            if df.empty:
                continue
            df = df[df["matched"] == True]
            for obj_uri in df.obj_uri.unique():
                df_by_obj = df[df["obj_uri"]==obj_uri]
                uuids = df_by_obj.uuid.tolist()
                sents = df_by_obj.match_sent.tolist()
                if obj_uri not in uri2sents:
                    uri2sents[obj_uri] = {}
                for uuid, sent in zip(uuids, sents):
                    if uuid not in uri2sents[obj_uri]:
                        uri2sents[obj_uri].update({uuid: {}})
                    uri2sents[obj_uri][uuid].update({lang: sent})
        return uri2sents

    def _get_match_sentences(self, lang, rel, match_type="partial-match"):
        if match_type == "full-match":
            dump_file = os.path.join(
                EVALUATION_ROOT[self.dataset.model_name],
                "matched_sentences",
                rel,
                f"{lang}-full-match.csv",
            )
        elif match_type == "partial-match":
            dump_file = os.path.join(
                EVALUATION_ROOT[self.dataset.model_name],
                "matched_sentences",
                rel,
                f"{lang}-partial-match.csv",
            )
        else:
            raise NotImplementedError(f"Matching type {match_type} is not avaliable")

        return pd.read_csv(dump_file) if os.path.exists(dump_file) else pd.DataFrame()

if __name__ == "__main__":
    from mask_dataset import MaskedDataset

    dataset = MaskedDataset(model_name="xlmr")
    generate_match_sentences(dataset)
    # prober = SentIterator(dataset)

    # matches = prober.get_correct_matches(rel="P103")
    