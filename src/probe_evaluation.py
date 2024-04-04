from locale import normalize
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langcodes import Language
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

from constants import PROBER_ANALYSIS_ROOT

def set_color_set():
    names = sorted(mcolors.CSS4_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c)))) # type: ignore
    for name in ['whitesmoke', 'white','snow']:
        names.remove(name)
    names = [names[i] for i in range(0, len(names)-1, len(names) // 53 + 1)]
    random.shuffle(names)
    return names

def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def get_layerwise_corr(lang1, lang2, layer_num, probing_result, normalization, label=None):
    try:
        shape = probing_result[lang1][2]
        if label is not None:
            summation_per_obj_lang1 = np.divide(probing_result[lang1][3], normalization, out=np.zeros_like(shape), where=normalization!=0)
            summation_per_obj_lang2 = np.divide(probing_result[lang2][3], normalization, out=np.zeros_like(shape), where=normalization!=0)
            summation_obj1 = summation_per_obj_lang1[label].reshape(layer_num, -1).astype("float32")
            summation_obj2 = summation_per_obj_lang2[label].reshape(layer_num, -1).astype("float32")
        else:
            summation_overall_obj_lang1 = np.divide(probing_result[lang1][2], normalization, out=np.zeros_like(shape), where=normalization!=0)
            summation_overall_obj_lang2 = np.divide(probing_result[lang2][2], normalization, out=np.zeros_like(shape), where=normalization!=0)
            summation_obj1 = summation_overall_obj_lang1.reshape(layer_num, -1).astype("float32")
            summation_obj2 = summation_overall_obj_lang2.reshape(layer_num, -1).astype("float32")

    except Exception as e:
        raise ValueError(f"{lang1}, {lang2}", e) from e
    return [
        euclidean_distance(summation_obj1[layer], summation_obj2[layer])
        for layer in range(len(summation_obj1))
    ]

def draw_single_line(x, y, xlabels, title, save_path=None):
    plt.plot(x, y)
    plt.title(title, fontsize = 12)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def draw_multiple_lines(x, ys, line_labels, title, save_path=None):
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', set_color_set())
    for y in ys:
        plt.plot(x, y)
    
    plt.title(title, fontsize = 12)
    plt.legend(line_labels, ncol=2, bbox_to_anchor=(1.0, 0.5), loc='center left')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def draw_active_neuron_heatmap(probing_score, layer_num, label_name=None, save_path=None, chunkize="mean", reload=False):
    """Draw distribution of active neurons from the probing score
    Args:
        probing_score: np.ndarray [layer_num, neuron_each_layer]
    """
    if save_path is not None and os.path.exists(save_path) and reload==False:
        return
    if probing_score is not None:
        if chunkize == 'mean':
            probing_score = probing_score.reshape(layer_num, -1, 16).mean(axis=1) # type: ignore
        elif chunkize == 'max':
            probing_score = probing_score.reshape(layer_num, -1, 16).max(axis=1) # type: ignore
        else:
            raise ValueError(f"Unsupported chunk method - {chunkize}")
        ax = sns.heatmap(probing_score, cmap="GnBu")
        ax.invert_yaxis()
        ax.set_xlabel('Neuron Blocks')
        ax.set_ylabel('Transformer Layers')
        if label_name is not None:
            title = f"Active Neuron Distribution Across Transform Layers - {label_name}"
        else:
            title = "Active Neuron Distribution Across Transform Layers"

        ax.set_title(title, fontsize=10)
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
"""Generate active neuron distance distribution acorss objects for all languages
"""
def get_no_match_labels(save_root):
    no_matches_file = os.path.join(save_root, "no_matching.txt")
    if os.path.exists(no_matches_file):
        with open(no_matches_file, 'r') as fp:
            return fp.readlines()
    else:
        with open(no_matches_file, 'w') as fp:
            return []
    
def is_no_match(obj, match_type, no_match_labels):
    return f"{obj}_{match_type}" in no_match_labels

def add_no_match_labels(save_root, lable, match_type):
    no_matches_file = os.path.join(save_root, "no_matching.txt")
    with open(no_matches_file, 'a+') as fp:
        return fp.write(f"{lable}_{match_type}\n")

"""Generating the distance of object-based active neurons between languages and draw the distance distribution across layers.
"""  
def generate_obj_overall_active_neuron_dist_across_langs(dataset, prober, rel, normalize=False, langs_set=None, probing_type='outs', display=False, reprobe=False, reload=False):
    _, probing_result = prober.probe_objs_per_lang(rel=rel, probing_type=probing_type, reload=reprobe)

    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "overall-obj-active-neurons-comparing-across-languages", rel)
    os.makedirs(save_root, exist_ok=True) # type: ignore

    fig_save_path = os.path.join(save_root, f"{probing_type}_{prober.match_type}.png")
    if reload==False and display==False and fig_save_path is not None and os.path.exists(fig_save_path) and reload==False:
        return

    corres = []
    matched_langs = dataset.get_langs_in_rel(rel)    
    
    if normalize:
        normalization = np.zeros_like(probing_result['en'][2], dtype='float32')
        for lang in matched_langs:
            if probing_result[lang][2] is None:
                continue
            normalization += probing_result[lang][2]
        normalization /= len(matched_langs)
    else:
        normalization = np.ones_like(probing_result['en'][2], dtype='float32')


    if langs_set is not None:
        matched_langs = langs_set

    pivot_lang = 'en'
    matched_langs.remove(pivot_lang)
    updated_matched_langs = []

    for lang in matched_langs:
        if probing_result[lang][2] is None:
            continue
        corr = get_layerwise_corr(pivot_lang, lang, prober.model.layer_num, probing_result, normalization=normalization) # type: ignore
        corres.append(corr)
        updated_matched_langs.append(lang)

    if display:
        fig_save_path = None
    draw_multiple_lines(
        list(range(len(corres[0]))),
        corres,
        [dataset.display_lang(lang) for lang in updated_matched_langs],
        f"Layer-wise & normalized distances of object-based active neurons \nRelation Template: {dataset.display_rel(rel)} \nBaseline language: {dataset.display_lang(pivot_lang)} | Neuron type: {probing_type} ",
        save_path=fig_save_path)

def generate_obj_active_neuron_dist_across_langs(
        dataset, match_type, rel_template, obj, 
        save_root, layer_num, obj2matchedlangs, 
        probing_result, probing_type='outs', 
        reload=False, normalize=False, display=False
    ):
    fig_save_path = os.path.join(save_root, f"{obj}_{probing_type}_{match_type}.png")
    if fig_save_path is not None and os.path.exists(fig_save_path) and reload==False:
        return
    
    if obj not in obj2matchedlangs or not obj2matchedlangs[obj]:
        add_no_match_labels(save_root, obj, match_type)
        return
    
    corres = []
    matched_langs = obj2matchedlangs[obj]
    
    baseline_lang = 'en'
    if baseline_lang not in matched_langs:
        baseline_lang = matched_langs[0]

    matched_langs.remove(baseline_lang)
    if not matched_langs:
        add_no_match_labels(save_root, obj, match_type)
        return
    
    if dataset.model_name == "xlmr":
        layer_num = 24
    elif dataset.model_name == "mbert":
        layer_num = 12
    
    normalization = np.ones_like(probing_result['en'][2], dtype='float32')
    if normalize:
        for lang in matched_langs:
            normalization += probing_result[lang][2]
        normalization /= len(matched_langs)

    for lang in matched_langs:
        corr = get_layerwise_corr(baseline_lang, lang, layer_num, probing_result, label=obj, normalization=normalization) # type: ignore
        corres.append(corr)
    
    if display:
        fig_save_path = None
    
    draw_multiple_lines(
        list(range(len(corres[0]))),
        corres,
        [Language.get(lang).display_name() for lang in matched_langs if lang != 'en'],
        f"Layer-wise active neuron distances for object ({dataset.display_obj(obj)}) \nRelation Template: {rel_template} \nBaseline language: {dataset.display_lang(baseline_lang)} | Neuron type: {probing_type} ",
        save_path=fig_save_path)

def generate_obj_active_neuron_dist(dataset, prober, rel, probing_type='outs', reprobe=False, reload=False, thread=1):
    objs_in_rel = dataset.get_objs_in_rel(rel)
    obj2matchedlangs, probing_result = prober.probe_objs_per_lang(rel=rel, probing_type=probing_type, reload=reprobe)

    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "obj-active-neurons-comparing-across-languages", rel)
    os.makedirs(save_root, exist_ok=True) # type: ignore
    no_match_objs = get_no_match_labels(save_root)

    if thread > 1:
        with ThreadPoolExecutor(max_workers=thread) as executor:
            futures = []
            for obj in objs_in_rel:
                if is_no_match(obj, prober.match_type, no_match_objs):
                    continue
                futures.append(
                    executor.submit(
                        generate_obj_active_neuron_dist_across_langs, 
                        dataset, prober.match_type, dataset.display_rel(rel),
                        obj, save_root, prober.model.layer_num, obj2matchedlangs, 
                        probing_result, probing_type, reload, False, False))
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating neuron distri for objs in {rel}"):
                future.result()
    elif thread == 1:
        for obj in tqdm(objs_in_rel, desc=f"Generating neuron distri for objs in {rel}"):
            if is_no_match(obj, prober.match_type, no_match_objs):
                continue
            generate_obj_active_neuron_dist_across_langs(
                dataset=dataset,
                match_type=prober.match_type,
                probing_type=probing_type,
                layer_num=prober.model.layer_num,
                obj=obj,
                obj2matchedlangs=obj2matchedlangs,
                probing_result=probing_result,
                save_root=save_root,
                reload=reload,
                rel_template=dataset.display_rel(rel),
                normalize=False,
                display=False
            )

def generate_obj_active_neuron_dist_for_all_rel(model_name, device="cuda:1", per_obj=False, reload=False):
    from prober import Prober
    from mask_dataset import MaskedDataset
    from constants import MATCH_TYPES, PROBE_TYPES
    
    dataset = MaskedDataset(model_name=model_name)
    for match_type in MATCH_TYPES:
    # for match_type in ['partial-match']:
        prober = Prober(dataset, match_type=match_type, device=device)
        for probe_type in PROBE_TYPES:
            for rel in dataset.rels:
            # for rel in ['P37']:
                if per_obj:
                    generate_obj_active_neuron_dist(dataset, prober, rel, probing_type=probe_type, reprobe=False, reload=reload, thread=56)
                else:
                    generate_obj_overall_active_neuron_dist_across_langs(dataset, prober, rel, probing_type=probe_type, reprobe=False, reload=reload)

"""Generate heatmap of objects active neurons per language
"""
def generate_obj_active_neuron_heatmap_across_languages(
        dataset, prober, rel, probing_type='outs', 
        reprobe=False, reload=False):    
    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "object-active-neurons-comparing-per-language", rel)
    os.makedirs(save_root, exist_ok=True)

    _, probing_result = prober.probe_objs_per_lang(rel=rel, probing_type=probing_type, reload=reprobe) # type: ignore

    langs_in_rel = dataset.get_langs_in_rel(rel)
    for lang in tqdm(langs_in_rel, desc=f"Generating heatmap of object active neurons for relation: {rel} - probe_type: {probing_type} - match_type: {prober.match_type}"):
        differ = probing_result[lang][2]
        if differ is None:
            continue

        fig_save_path = os.path.join(save_root, f"{lang}_{prober.match_type}_{probing_type}.png")
        draw_active_neuron_heatmap(
            probing_score=probing_result[lang][2], 
            label_name=f"Language ({dataset.display_lang(lang)})",
            layer_num=prober.model.layer_num, 
            save_path=fig_save_path,
            reload=reload)

"""Generate heatmap of objects active neurons mixing all languages together
"""
def generate_obj_active_neuron_heatmap(
    dataset, prober, rel, probing_type='outs', 
    reprobe=False, reload=False):    

    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "object-active-neurons-comparing-ignoring-languages", rel)
    os.makedirs(save_root, exist_ok=True)

    probing_result = prober.probe_objs(rel=rel, probing_type=probing_type, reload=reprobe) # type: ignore

    fig_save_path = os.path.join(save_root, f"{rel}_{prober.match_type}_{probing_type}.png")
    draw_active_neuron_heatmap(
        probing_score=probing_result[2], 
        layer_num=prober.model.layer_num, 
        save_path=fig_save_path,
        reload=reload)

"""Generate heatmap of language active neurons per object
"""
def generate_lang_active_neuron_heatmap_across_objects(
        dataset, prober, rel, probing_type='outs', 
        reprobe=False, reload=False):    
    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "language-active-neurons-comparing-per-object", rel)
    os.makedirs(save_root, exist_ok=True)

    _, probing_result = prober.probe_langs_per_obj(rel=rel, probing_type=probing_type, reload=reprobe) # type: ignore

    objs_in_rel = dataset.get_objs_in_rel(rel)
    for obj in tqdm(objs_in_rel, desc=f"Generating heatmap of language active neurons per object for relation: {rel} - probe_type: {probing_type} - match_type: {prober.match_type}"):
        differ = probing_result[obj][2]
        if differ is None:
            continue

        fig_save_path = os.path.join(save_root, f"{obj}_{prober.match_type}_{probing_type}.png")
        draw_active_neuron_heatmap(
            probing_score=probing_result[obj][2], 
            label_name=f"Object ({dataset.display_obj(obj)})",
            layer_num=prober.model.layer_num, 
            save_path=fig_save_path,
            reload=reload)

"""Generate heatmap of language active neurons mixing all objects together
"""
def generate_lang_active_neuron_heatmap(
    dataset, prober, rel, probing_type='outs', 
    reprobe=False, reload=False):    

    save_root = os.path.join(PROBER_ANALYSIS_ROOT[dataset.model_name], "language-active-neurons-comparing-ignoring-objects", rel)
    os.makedirs(save_root, exist_ok=True)

    probing_result = prober.probe_langs(rel=rel, probing_type=probing_type, reload=reprobe) # type: ignore

    fig_save_path = os.path.join(save_root, f"{rel}_{prober.match_type}_{probing_type}.png")
    draw_active_neuron_heatmap(
        probing_score=probing_result[2], 
        layer_num=prober.model.layer_num, 
        save_path=fig_save_path,
        reload=reload)

def generate_active_neuron_heatmaps(model_name, device='cuda:0', active_type="object", per_label=True, probe_types=['acts'], reload=False):
    from prober import Prober
    from mask_dataset import MaskedDataset
    # print("per_label = ", per_label)
    dataset = MaskedDataset(model_name=model_name, reload=False)
    # for match_type in MATCH_TYPES:
    for match_type in ['full-match']:
        prober = Prober(dataset, match_type=match_type, device=device)    
        for rel in dataset.rels:
            print(f"Start to generate language active neuron distritutions for relation:{rel} | per_obj: {per_label}")
            for probe_type in probe_types:
                if active_type == "language":
                    if per_label:
                        generate_lang_active_neuron_heatmap_across_objects(dataset, prober, rel, probe_type, reload=reload)
                    else:
                        generate_lang_active_neuron_heatmap(dataset, prober, rel, probe_type, reload=reload)
                elif active_type == "object":
                    if per_label:
                        generate_obj_active_neuron_heatmap_across_languages(dataset, prober, rel, probe_type, reload=reload)
                    else:
                        generate_obj_active_neuron_heatmap(dataset, prober, rel, probe_type, reload=reload)
                else:
                    raise NotImplementedError(f"Active type {active_type} is not supported")
                    

if __name__ == "__main__":
    # from mask_dataset import MaskedDataset
    # from prober import Prober
    # from sents_iterator import SentIterator
    
    # sent_iter = SentIterator(dataset)
    
    # dataset = MaskedDataset(model_name="mbert")
    # rels_info = dataset.get_rel_info()
    # prober = Prober(dataset, match_type='full-match', device='cuda:1')
    # for rel in rels_info:
    #     print(f"Start to process relation - {rel} - {rels_info[rel]['en']}")
    #     generate_obj_active_neuron_dist(dataset, prober, rel, probing_type='acts', reprobe=True, reload=True, thread=40)
    #     generate_obj_active_neuron_dist(dataset, prober, rel, probing_type='outs', reprobe=True, reload=True, thread=40)
    

    # from utils import ResourcePool
    # class ProbingEvalutionPool(ResourcePool):
    #     def task(self, args):
    #         idx, prober = self.assign()
    #         rel = args
    #         print(f"Start to process relation - {rel} - {rels_info[rel]['en']}")
    #         generate_obj_active_neuron_dist(dataset, prober, rel, probing_type='acts', reprobe=False, thread=20)
    #         generate_obj_active_neuron_dist(dataset, prober, rel, probing_type='outs', reprobe=False, thread=20)
    #         # print("Thread finish running prediction task for {}-{}, with {}-th {} model".format(lang, rel, idx, model.name))
    #         self.release(idx)

    # thread = 7
    # resources = [Prober(dataset, match_type='partial-match', device="cuda:{}".format(idx)) for idx in tqdm(range(thread), desc="Loading probers for parallel processing")]
    # probing_evaluator_pool = ProbingEvalutionPool(resources, thread)
    # probing_evaluator_pool.run(iterator=rels_info.keys())

    # generate_lang_active_neuron_dist_for_all_rel('mbert', active_type="object", per_label=False)

    # from mask_dataset import MaskedDataset
    # from sents_iterator import SentIterator
    # from prober import Prober

    # dataset = MaskedDataset(model_name="mbert", reload=False)
    # sent_iter = SentIterator(dataset)
    # prober = Prober(dataset, device='cuda:2')

    # rel = 'P37'
    # from probe_evaluation import generate_obj_active_neuron_dist_for_all_rel
    # generate_obj_active_neuron_dist_for_all_rel('mbert', reload=True)

    generate_active_neuron_heatmaps('mbert', device="cuda:1", active_type="object", per_label=False, reload=True)