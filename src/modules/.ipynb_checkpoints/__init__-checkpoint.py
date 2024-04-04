from modules.load_xlm_mlm import load_xlm_mlm
from modules.load_xlmr import load_xlmr_base, collect_neurons_from_xlmr

MODEL_LOADING_FUNCTIONS = {
    "xlm_mlm": load_xlm_mlm,
    "xlmr_base": load_xlmr_base
}

NEURON_COLLECTOR = {
    "xlmr_base": collect_neurons_from_xlmr
}