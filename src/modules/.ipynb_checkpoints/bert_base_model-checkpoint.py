import torch
from tqdm import tqdm
import torch.distributed as dist
from modules.abstract_model import AbstractModel
from transformers import BertTokenizer, BertForMaskedLM

class BERTBaseModel(AbstractModel):
    def __init__(self, device):
        super().__init__(device)
        self.mask_token = "[MASK]"
        self.name = "mbert"
    
    def _load_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased").to(self.device)
        return tokenizer, model
    
    def tokens_to_ids(self, objs):
        return self.tokenizer.convert_tokens_to_ids(objs)
    
    def token_to_id(self, obj):
        return self.tokenizer._convert_token_to_id(obj)
    
    def ids_to_tokens(self, objs):
        return self.tokenizer.convert_ids_to_tokens(objs)
    
    def id_to_token(self, obj):
        return self.tokenizer._convert_id_to_token(obj)
    
    def collect_neurons(self, sentences):
        pass

    def collect_topk_pred(self, sentences, topk, logits):
        pass
    