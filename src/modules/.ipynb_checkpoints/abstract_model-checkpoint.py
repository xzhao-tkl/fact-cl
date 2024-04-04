import torch
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, device):
        self.device = device
        self.is_cuda = "cuda" in self.device.type
        self.tokenizer, self.model = self._load_model()
        self.mask_token = None

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def collect_neurons(self, sentences):
        pass

    @abstractmethod
    def collect_topk_pred(self, sentences, topk, logits):
        pass
    
    @abstractmethod
    def tokens_to_ids(self, objs):
        pass
    
    @abstractmethod
    def token_to_id(self, obj):
        pass
    
    @abstractmethod
    def ids_to_tokens(self, objs):
        pass
    
    @abstractmethod
    def id_to_token(self, obj):
        pass
    
    def get_mask_tokens(self, sentences, mask_num):
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded_inputs).logits

        preds = []
        for i, sent in enumerate(sentences):
            masked_tokens = []
            tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
            mask_ind = tokens.index(self.mask_token)
            for j in range(mask_num):
                token = self.ids_to_tokens(torch.topk(logits[i][mask_ind+j], 1).indices)
                masked_tokens.append(token)
            preds.append(masked_tokens)
        return preds  
        