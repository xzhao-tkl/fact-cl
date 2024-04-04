import torch
from torch import nn
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel, apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.models.xlm.modeling_xlm import TransformerFFN, get_masks


class TransformerFFNProbing(TransformerFFN):
    def __init__(self, *args):
        if type(args[0]) is TransformerFFN:
            self.__dict__ = args[0].__dict__.copy()

    def get_activation(self, input):
        return apply_chunking_to_forward(self.act_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def act_chunk(self, input):
        x = self.lin1(input)
        return self.act(x)


class XLMWithLMHeadModelProbing(XLMWithLMHeadModel):
    def __init__(self, parent):
        self.__dict__ = parent.__dict__

    def forward(self,
                input_ids=None,
                attention_mask=None,
                langs=None,
                token_type_ids=None,
                position_ids=None,
                lengths=None,
                cache=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ffn_activation = transformer_outputs[0]
        ffn_outputs = transformer_outputs[1]
        output = transformer_outputs[2][0]
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return ffn_activation, ffn_outputs, MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs[2].hidden_states,
            attentions=transformer_outputs[2].attentions,
        )


class XLMModelProbing(XLMModel):
    def __init__(self, parent):
        self.__dict__ = parent.__dict__

    def forward(self,
                input_ids,
                attention_mask=None,
                langs=None,
                token_type_ids=None,
                position_ids=None,
                lengths=None,
                cache=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            bs, slen = input_ids.size()
        else:
            bs, slen = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.tensor([slen] * bs, device=device)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :slen]
        else:
            assert position_ids.size() == (bs, slen)  # (slen, bs)

        # langs
        if langs is not None:
            assert langs.size() == (bs, slen)  # (slen, bs)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        ffn_activations = []
        ffn_outputs = []
        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                cache=cache,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            ffn_activations.append(self.ffns[i].get_activation(tensor))
            ffn_output = self.ffns[i](tensor)
            ffn_outputs.append(ffn_output)
            tensor = tensor + ffn_output
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update .cache length
        if cache is not None:
            cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)
        return [torch.stack(ffn_activations, dim=0),
                torch.stack(ffn_outputs, dim=0),
                BaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)]


def load_xlm_mlm(device):
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-xnli15-1024")
    model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-xnli15-1024").to(device)

    # input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")], device=device)
    # language_id = tokenizer.lang2id["en"]
    # langs = torch.tensor([language_id] * input_ids.shape[1], device=device)
    # langs = langs.view(1, -1)

    new_ffns = nn.ModuleList()
    for layer in model.transformer.ffns:
        new_ffns.append(TransformerFFNProbing(layer))
    model.transformer.ffns = new_ffns
    model.transformer = XLMModelProbing(model.transformer)
    model = XLMWithLMHeadModelProbing(model)

    return tokenizer, model



MODEL_LOADING_FUNCTIONS = {
    "xlm_mlm": load_xlm_mlm
}
