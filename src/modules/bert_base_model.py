import os
import sys
import warnings
from sklearn.utils import deprecated

import torch
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from modules.abstract_model import AbstractModel

MAX_LAYER = 12

def intervene_neurons(hidden_states: torch.Tensor, mask_indice: torch.Tensor, neuron_mask: torch.Tensor, obj_embedding: torch.Tensor) -> torch.Tensor:
    """ Set the value of masked neurons to zero
    Args:
        hidden_states (tensor): batch_size x token_cnt x hidden_size
        mask_indice (tensor): batch_size x token_cnt, 0/1 tensor
        neuron_mask (tensor): hidden_size, 0/1 tensor
        obj_embedding (tensor): hidden_size, embedding of base languages
    Returns:
        hidden_states
    """
    neuron_mask = (1 - mask_indice.unsqueeze(2)) * torch.ones(neuron_mask.shape).to(neuron_mask.device) + mask_indice.unsqueeze(2) * neuron_mask
    return hidden_states * neuron_mask

@deprecated # type: ignore
def suppress_neurons(hidden_states: torch.Tensor, mask_indice: torch.Tensor, neuron_mask: torch.Tensor, obj_embedding: torch.Tensor) -> torch.Tensor:
    """ Set the value of masked neurons to zero
    Args:
        hidden_states (tensor): batch_size x token_cnt x hidden_size
        mask_indice (tensor): batch_size x token_cnt, 0/1 tensor
        neuron_mask (tensor): hidden_size, 0/1 tensor

    Returns:
        hidden_states
    """
    prev_shape = hidden_states.shape
    prev_dtype = hidden_states.dtype
    
    mask = mask_indice.unsqueeze(2) * neuron_mask
    hidden_states = hidden_states * (1 - mask)
    
    if prev_shape != hidden_states.shape:
        raise IndexError(f"The previous hidden state has the shape {prev_shape} but got {hidden_states.shape} after neuron suppression")
    if prev_dtype != hidden_states.dtype:
        raise IndexError(f"The previous hidden state has the type of {prev_dtype} but got {hidden_states.dtype} after neuron suppression. The neuron_mask type is {neuron_mask.dtype}")

    return hidden_states
    
@deprecated # type: ignore
def align_neurons(hidden_states: torch.Tensor, mask_indice: torch.Tensor, neuron_mask: torch.Tensor, srclang_obj_embedding: torch.Tensor) -> torch.Tensor:
    """ Set the value of masked neurons to maximum values in the hidden_states
    Args:
        hidden_states (tensor): batch_size x hidden_size
        neuron_mask (tensor): hidden_size, 0/1 tensor

    Returns:
        hidden_states
    """
    prev_shape = hidden_states.shape
    prev_dtype = hidden_states.dtype
    
    mask = mask_indice.unsqueeze(2) * neuron_mask
    # Replace the active neuron value by values in the soruce language
    hidden_states = hidden_states * (1 - mask) + srclang_obj_embedding * neuron_mask
    
    if prev_shape != hidden_states.shape:
        raise IndexError(f"The previous hidden state has the shape {prev_shape} but got {hidden_states.shape} after neuron suppression")
    if prev_dtype != hidden_states.dtype:
        raise IndexError(f"The previous hidden state has the type of {prev_dtype} but got {hidden_states.dtype} after neuron suppression. The neuron_mask type is {neuron_mask.dtype}")

    neuron_mask = mask_indice.unsqueeze(2) * neuron_mask
    return hidden_states * neuron_mask

@deprecated # type: ignore
def amplify_neurons(hidden_states: torch.Tensor, mask_indice: torch.Tensor, neuron_mask: torch.Tensor, srclang_obj_embedding: torch.Tensor) -> torch.Tensor:
    """ Set the value of masked neurons to maximum values in the hidden_states
    Args:
        hidden_states (tensor): batch_size x hidden_size
        neuron_mask (tensor): hidden_size, 0/1 tensor

    Returns:
        hidden_states
    """
    neuron_mask = mask_indice.unsqueeze(2) * neuron_mask
    # Replace the active neuron value by values in the soruce language
    hidden_states = hidden_states * neuron_mask
    return hidden_states * neuron_mask

class ActCollector(BertIntermediate):
    def __init__(self, *args, **kargs):
        if type(args[0]) is BertIntermediate:
            self.__dict__ = args[0].__dict__.copy()
        self.act_buffer = kargs["act_buffer"]

    def forward(self, hidden_states):
        hidden_states = super().forward(hidden_states)
        self.act_buffer.append(hidden_states.cpu())
        return hidden_states

class OutCollector(BertOutput):
    def __init__(self, *args, **kargs):
        if type(args[0]) is BertOutput:
            self.__dict__ = args[0].__dict__.copy()
        self.out_buffer = kargs["out_buffer"]

    def forward(self, hidden_states, input_tensor):
        hidden_states = super().forward(hidden_states, input_tensor)
        self.out_buffer.append(hidden_states.cpu())
        return hidden_states

class ActInterventor(BertIntermediate):
    def __init__(self, *args, **kargs):
        if type(args[0]) is BertIntermediate:
            self.__dict__ = args[0].__dict__.copy()
        self.layer_id = kargs["layer_id"]
        self.intervent_res_queue = kargs["intervent_res_queue"]
        self.mask_inidce_queue = kargs["mask_inidce_queue"]
        
    def forward(self, hidden_states):
        neuron_mask, obj_embedding = self.intervent_res_queue.popleft()
        if self.layer_id < MAX_LAYER - 1:
            mask_indice = self.mask_inidce_queue[0]
        else:
            mask_indice = self.mask_inidce_queue.popleft()
        if len(self.intervent_res_queue) != 0:
            raise IndexError(f"The intervent_res_queue need to be empty after consumed, but get {len(self.intervent_res_queue)} left")
        hidden_states = super().forward(hidden_states)
        return intervene_neurons(hidden_states, mask_indice, neuron_mask, obj_embedding)

class OutInterventor(BertOutput):
    def __init__(self, *args, **kargs):
        if type(args[0]) is BertOutput:
            self.__dict__ = args[0].__dict__.copy()
        self.layer_id = kargs["layer_id"]
        self.intervent_res_queue = kargs["intervent_res_queue"]
        self.mask_inidce_queue = kargs["mask_inidce_queue"]

    def forward(self, hidden_states, input_tensor):
        neuron_mask, obj_embedding = self.intervent_res_queue.popleft()
        if self.layer_id < MAX_LAYER - 1:
            mask_indice = self.mask_inidce_queue[0]
        else:
            mask_indice = self.mask_inidce_queue.popleft()
        
        if len(self.intervent_res_queue) != 0:
            raise ValueError(f"The intervent_res_queue need to be empty after consumed, but get {len(self.intervent_res_queue)} left")
        
        hidden_states = super().forward(hidden_states, input_tensor)
        return intervene_neurons(hidden_states, mask_indice, neuron_mask, obj_embedding)
    
class BERTBaseModel(AbstractModel):
    def __init__(self, device, intervened=False, intervened_neuron_type='acts', collect_mode=True):
        """
        Args:
            device (_type_)
            intervened (bool)
        """
        super().__init__(
            device=device, 
            name="mbert", 
            mask_token="[MASK]",
            layer_num=12,
            intervened=intervened,
            collect_mode=collect_mode,
            intervened_neuron_type=intervened_neuron_type)

    def _load_model(self):
        warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at bert-base-multilingual-cased")
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased").to(self.device) # type: ignore
        for layer, ffn_layer in enumerate(model.bert.encoder.layer):
            if not self.intervened:
                if self.collect_mode:
                    ffn_layer.intermediate = ActCollector(
                        ffn_layer.intermediate, 
                        act_buffer=self.act_buffer).to(self.device)
                    ffn_layer.output = OutCollector(
                        ffn_layer.output,
                        out_buffer=self.out_buffer).to(self.device)
            else:
                if self.intervened_neuron_type == 'acts':
                    ffn_layer.intermediate = ActInterventor(
                        ffn_layer.intermediate, 
                        layer_id=layer,
                        intervent_res_queue=self.neuron_mask_queues[layer],
                        mask_inidce_queue=self.mask_inidce_queue).to(self.device)
                elif self.intervened_neuron_type == 'outs':
                    ffn_layer.output = OutInterventor(
                        ffn_layer.output, 
                        layer_id=layer,
                        intervent_res_queue=self.neuron_mask_queues[layer],
                        mask_inidce_queue=self.mask_inidce_queue).to(self.device)    
        return tokenizer, model
    