import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, XLMRobertaForMaskedLM
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaIntermediate, XLMRobertaOutput)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from modules.abstract_model import AbstractModel

class MyXLMRobertaIntermediate(XLMRobertaIntermediate):
    def __init__(self, *args, **kargs):
        if type(args[0]) is XLMRobertaIntermediate:
            self.__dict__ = args[0].__dict__.copy()
        self.act_buffer = kargs["act_buffer"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        self.act_buffer.append(hidden_states.cpu())
        return hidden_states


class MyXLMRobertaOutput(XLMRobertaOutput):
    def __init__(self, *args, **kargs):
        if type(args[0]) is XLMRobertaOutput:
            self.__dict__ = args[0].__dict__.copy()
        self.out_buffer = kargs["out_buffer"]

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, input_tensor)
        self.out_buffer.append(hidden_states.cpu())
        return hidden_states


class XLMBaseModel(AbstractModel):
    def __init__(self, device, intervened=False, intervened_neuron_type='acts', collect_mode=True):
        super().__init__(
            device=device, 
            name="xlmr", 
            mask_token="<mask>", 
            layer_num=24, 
            intervened=intervened,
            collect_mode=collect_mode,
            intervened_neuron_type=intervened_neuron_type)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large").to(self.device)
        if self.collect_mode:
            for ffn_layer in model.roberta.encoder.layer:
                ffn_layer.intermediate = MyXLMRobertaIntermediate(
                    ffn_layer.intermediate, 
                    act_buffer=self.act_buffer).to(self.device)
                ffn_layer.output = MyXLMRobertaOutput(
                    ffn_layer.output,
                    out_buffer=self.out_buffer).to(self.device)
        return tokenizer, model

if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = XLMBaseModel(device)
