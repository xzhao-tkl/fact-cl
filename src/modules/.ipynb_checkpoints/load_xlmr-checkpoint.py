import torch
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaIntermediate, XLMRobertaOutput
from transformers import AutoTokenizer, XLMRobertaForMaskedLM

ActivationReceiver = []
LayeroutputReceiver = []


class MyXLMRobertaIntermediate(XLMRobertaIntermediate):
    def __init__(self, *args):
        if type(args[0]) is XLMRobertaIntermediate:
            self.__dict__ = args[0].__dict__.copy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        ActivationReceiver.append(hidden_states)
        return hidden_states


class MyXLMRobertaOutput(XLMRobertaOutput):
    def __init__(self, *args):
        if type(args[0]) is XLMRobertaOutput:
            self.__dict__ = args[0].__dict__.copy()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, input_tensor)
        LayeroutputReceiver.append(hidden_states)
        return hidden_states


def load_xlmr_base(device):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
    model = model.to(device)
    for ffn_layer in model.roberta.encoder.layer:
        ffn_layer.intermediate = MyXLMRobertaIntermediate(ffn_layer.intermediate)
        ffn_layer.output = MyXLMRobertaOutput(ffn_layer.output)
    return tokenizer, model


def collect_neurons_from_xlmr(tokenizer, model, sentence, device):
    global ActivationReceiver, LayeroutputReceiver
    assert len(ActivationReceiver) == len(LayeroutputReceiver) == 0
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    act = torch.stack(ActivationReceiver)
    out = torch.stack(LayeroutputReceiver)
    ActivationReceiver = []
    LayeroutputReceiver = []

    return logits, act, out

if __name__ == '__main__':
    tokenizer, model = load_xlmr_base()
    sentence = "Pairs is the capital of <mask>"
    logits, act, out = collect_neurons_from_xlmr(tokenizer, model, sentence)
    print(act.shape, out.shape)
    print(len(ActivationReceiver), len(LayeroutputReceiver))
