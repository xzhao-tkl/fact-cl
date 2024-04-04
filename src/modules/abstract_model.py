import os
import sys
from typing import Optional
from collections import deque

from numpy import deprecate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from abc import ABC, abstractmethod

import torch

from src.utils import batchify


class AbstractModel(ABC):
    def __init__(self, device, name, mask_token, layer_num, intervened, collect_mode, intervened_neuron_type):
        self.device = device
        self.is_cuda = "cuda" in self.device.type
        self.mask_token = mask_token
        self.layer_num = layer_num
        self.intervened = intervened
        self.collect_mode = collect_mode
        self.intervened_neuron_type = intervened_neuron_type
        self.name = name
        
        if intervened_neuron_type not in ["acts", 'outs']:
            raise ValueError(f"Unsupported intervened_neuron_type - {intervened_neuron_type}")
        
        if intervened:
            self.neuron_mask_queues = [deque() for i in range(self.layer_num)]
            self.mask_inidce_queue = deque()
            self.name += f"-intervened"
        else:
            self.act_buffer = []
            self.out_buffer = []
        
        self.tokenizer, self.model = self._load_model() # type: ignore

    @abstractmethod
    def _load_model(self):
        pass
    
    def tokens_to_ids(self, objs):
        return self.tokenizer.convert_tokens_to_ids(objs)

    def ids_to_tokens(self, objs):
        return self.tokenizer.convert_ids_to_tokens(objs)

    def token_to_id(self, obj):
        return self.tokenizer._convert_token_to_id(obj)

    def id_to_token(self, obj):
        return self.tokenizer._convert_id_to_token(obj)

    def collect_topk_pred(self, sentences, topk, logits):
        topks = []
        for i in range(len(sentences)):
            sent = sentences[i]
            inputs = self.tokenizer(sent, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            try:
                mask_ind = tokens.index(self.mask_token)
            except ValueError as e:
                raise ValueError(f"'{self.mask_token}' is not in sentence {sent}") from e
            topks.append(
                self.tokenizer.convert_ids_to_tokens(
                    torch.topk(logits[i][0][mask_ind], topk).indices
                )
            )
        torch.cuda.empty_cache()
        return topks

    def collect_topk_pred_mltokens(self, sentences, topk, logits, mask_num):
        topks = []
        for i in range(len(sentences)):
            sent = sentences[i]
            inputs = self.tokenizer(sent, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            try:
                mask_ind = tokens.index(self.mask_token)
                mask_inds = [mask_ind + i for i in range(mask_num)]
            except ValueError as e:
                raise ValueError(f"'{self.mask_token}' is not in sentence {sent}") from e
            tokens = [
                self.tokenizer.convert_ids_to_tokens(
                    torch.topk(logits[i][0][ind], topk).indices
                )
                for ind in mask_inds
            ]
            topks.append(tokens)
        torch.cuda.empty_cache()
        return topks
    
    def indices(self, tokens):
        return [i for i, x in enumerate(tokens) if x == self.mask_token]
    
    @deprecate
    def get_mask_tokens_with_masknum(self, sentences, mask_num):
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded_inputs).logits

        preds = []
        for i in range(len(sentences)):
            masked_tokens = []
            tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
            mask_ind = tokens.index(self.mask_token)
            for j in range(mask_num):
                token = self.ids_to_tokens(torch.topk(logits[i][mask_ind+j], 1).indices)
                masked_tokens.append(token)
            preds.append(masked_tokens)
        return preds  
    
    def get_mask_tokens(self, sentences: list[str]) -> list[list[str]]:
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded_inputs).logits

        preds = []
        for i in range(len(sentences)):
            masked_tokens = []
            tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
            mask_ind = tokens.index(self.mask_token)
            for j, token in enumerate(tokens[mask_ind:]):
                if token != self.mask_token:
                    break
                token = self.ids_to_tokens(torch.topk(logits[i][mask_ind+j], 1).indices)
                masked_tokens.append(token)
            preds.append(masked_tokens)
        return preds  
    
    def get_mask_tokens_ids(self, sentences: list[str]) -> tuple[list[list[str]], list[list[int]]]:
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded_inputs).logits
            
        mask_tokens_ls = []
        mask_tokenids_ls = []
        for i in range(len(sentences)):
            masked_tokens = []
            masked_tokenids = []
            tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
            mask_ind = tokens.index(self.mask_token)
            for j, token in enumerate(tokens[mask_ind:]):
                if token != self.mask_token:
                    break
                tokenid = torch.topk(logits[i][mask_ind+j], 1).indices
                masked_tokens.append(self.ids_to_tokens(tokenid))
                masked_tokenids.append(tokenid.tolist())
            mask_tokens_ls.append(masked_tokens)
            mask_tokenids_ls.append(masked_tokenids)
        return mask_tokens_ls, mask_tokenids_ls
    
    # @neuron_loader
    def _collect_neurons(self, sentences, reload=False, verbose=False):
        if not self.intervened:
            assert len(self.act_buffer) == len(self.out_buffer) == 0

        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded_inputs).logits
        
        assert len(self.act_buffer) == len(self.out_buffer) == self.layer_num

        result = []
        for i, sent in enumerate(sentences):
            masked_tokens = []
            tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
            mask_inds = self.indices(tokens)
            for idx in mask_inds:
                token = self.ids_to_tokens(torch.topk(logits[i][idx], 1).indices)
                masked_tokens.append(token)
            neuron_info = {
                'sent': sent, 
                'mask_inds': mask_inds,
                'masked_tokens': masked_tokens, 
                'acts': torch.stack([self.act_buffer[layer_idx][i][mask_inds, :] for layer_idx in range(self.layer_num)]),
                'outs': torch.stack([self.out_buffer[layer_idx][i][mask_inds, :] for layer_idx in range(self.layer_num)])
            }
            result.append(neuron_info)

        self.act_buffer.clear()
        self.out_buffer.clear()
        return result

    def collect_neurons(self, sentences, batchsize=32, reload=False, verbose=False) -> list:
        if len(sentences) == 0:
            return []
        
        result = []
        for sent in list(batchify(sentences, batchsize)):
            result.extend(self._collect_neurons(sentences=sent, reload=reload, verbose=verbose))
        return result
    
    def inject_intervention_resources(self, neuron_masks: torch.Tensor, obj_embeddings: Optional[torch.Tensor]):
        """ Set neuron masks for each layer for the next forward processing

        Args:
            neuron_masks (torch.tensor): layer_num x layer_size
        """
        for layer in range(self.layer_num):
            if obj_embeddings is not None:
                self.neuron_mask_queues[layer].append((neuron_masks[layer], obj_embeddings[layer]))
            else:
                self.neuron_mask_queues[layer].append((neuron_masks[layer], None))

    def inject_mask_indices(self, mask_inidce: torch.Tensor):
        """ Set mask indices for each layer for the next forward processing
        """
        self.mask_inidce_queue.append(mask_inidce)

    def check_neuron_mask_queue_size(self):
        return [len(queue) for queue in self.neuron_mask_queues]