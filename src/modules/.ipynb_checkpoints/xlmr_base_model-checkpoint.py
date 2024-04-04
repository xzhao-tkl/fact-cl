import torch
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer, XLMRobertaForMaskedLM
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaIntermediate, XLMRobertaOutput
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.abstract_model import AbstractModel

ActBuffer = []
OutBuffer = []


class MyXLMRobertaIntermediate(XLMRobertaIntermediate):
    def __init__(self, *args):
        if type(args[0]) is XLMRobertaIntermediate:
            self.__dict__ = args[0].__dict__.copy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        ActBuffer.append(hidden_states)
        return hidden_states


class MyXLMRobertaOutput(XLMRobertaOutput):
    def __init__(self, *args):
        if type(args[0]) is XLMRobertaOutput:
            self.__dict__ = args[0].__dict__.copy()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, input_tensor)
        OutBuffer.append(hidden_states)
        return hidden_states


class XLMBaseModel(AbstractModel):
    def __init__(self, device):
        super().__init__(device)
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.unk_id = 3
        self.name = "xlmr"
        
    def _load_model(self):
        # dist.init_process_group(backend='nccl', init_method='env://')
        # torch.cuda.set_device(0)
        # torch.cuda.manual_seed_all(42)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large").to(self.device)
        # model = DDP(
        #     model,
        #     device_ids=[0],
        #     output_device=0
        # )
        # for ffn_layer in model.roberta.encoder.layer:
        #     ffn_layer.intermediate = MyXLMRobertaIntermediate(ffn_layer.intermediate)
        #     ffn_layer.output = MyXLMRobertaOutput(ffn_layer.output)
        return tokenizer, model
    
#     def get_mask_tokens(self, sentences, mask_num):
#         encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             logits = self.model(**encoded_inputs).logits

#         preds = []
#         for i, sent in enumerate(sentences):
#             masked_tokens = []
#             tokens = self.ids_to_tokens(encoded_inputs['input_ids'][i])
#             mask_ind = tokens.index(self.mask_token)
#             for j in range(mask_num):
#                 token = self.ids_to_tokens(torch.topk(logits[i][mask_ind+j], 1).indices)
#                 masked_tokens.append(token)
#             preds.append(masked_tokens)
#         return preds  
        
    
    def collect_neurons(self, sentences):
        global ActBuffer, OutBuffer
        assert len(ActBuffer) == len(OutBuffer) == 0
        logits = []
        acts = []
        outs = []
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits.append(self.model(**inputs).logits)
            if self.is_cuda:
                _act_gpu = torch.stack(ActBuffer)
                _out_gpu = torch.stack(OutBuffer)
                act = _act_gpu.cpu().numpy()
                out = _out_gpu.cpu().numpy()
                del _act_gpu, _out_gpu
                torch.cuda.empty_cache()
            else:
                act = torch.stack(ActBuffer)
                out = torch.stack(OutBuffer)

            acts.append(act)
            outs.append(out)
            ActBuffer = []
            OutBuffer = []
        return logits, acts, outs

    def collect_neurons_iter(self, sentences):
        global ActBuffer, OutBuffer
        assert len(ActBuffer) == len(OutBuffer) == 0
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logit = self.model(**inputs).logits
            if self.is_cuda:
                _act_gpu = torch.stack(ActBuffer)
                _out_gpu = torch.stack(OutBuffer)
                act = _act_gpu.cpu().numpy()
                out = _out_gpu.cpu().numpy()
                del _act_gpu, _out_gpu
                torch.cuda.empty_cache()
            else:
                act = torch.stack(ActBuffer)
                out = torch.stack(OutBuffer)
            ActBuffer = []
            OutBuffer = []
            yield logit, act, out

    def collect_topk_pred(self, sentences, topk, logits):
        topks = []
        for i in range(len(sentences)):
            sent = sentences[i]
            inputs = self.tokenizer(sent, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            try:
                mask_ind = tokens.index(self.mask_token)
            except ValueError as e:
                raise ValueError("'{}' is not in sentence {}".format(self.mask_token, sent))
            topks.append(self.tokenizer.convert_ids_to_tokens(torch.topk(logits[i][0][mask_ind], topk).indices))
        torch.cuda.empty_cache()
        return mask_ind, topks
    
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
                raise ValueError("'{}' is not in sentence {}".format(self.mask_token, sent))    
            tokens = [self.tokenizer.convert_ids_to_tokens(torch.topk(logits[i][0][ind], topk).indices) for ind in mask_inds]
            topks.append(tokens)
        torch.cuda.empty_cache()
        return mask_ind, topks
    
    def collect_obj_tokens(self, objs):
        obj_ids = self.tokenizer.convert_tokens_to_ids(objs)
        new_obj_idx = []
        for i, obj in enumerate(objs):
            if obj_ids[i] == self.unk_id:
                # print("object label {} not in model vocabulary".format(obj))
                continue
            new_obj_idx.append(i)
        return new_obj_idx
    
    def tokens_to_ids(self, objs):
        return self.tokenizer.convert_tokens_to_ids(objs)
    
    def ids_to_tokens(self, objs):
        return self.tokenizer.convert_ids_to_tokens(objs)
    
    def token_to_id(self, obj):
        return self.tokenizer._convert_token_to_id(obj)
    
    def id_to_token(self, obj):
        return self.tokenizer._convert_id_to_token(obj)
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = XLMBaseModel(device)
    