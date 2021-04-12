import torch
from torch.nn import functional as F
from torch import nn
import torchvision.models as vmodels

from transformers import *
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics as compute_metrics,
    glue_convert_examples_to_features as convert_examples_to_features,
    glue_output_modes as output_modes,
    glue_processors as processors,
)

LANG_MODELS = {
          'bert':    (BertModel,       BertTokenizer,       'bert-base-uncased'),
          'bert-large':  (BertModel,       BertTokenizer,       'bert-large-uncased'),
          'gpt':     (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          'gpt2':    (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          'ctrl':    (CTRLModel,       CTRLTokenizer,       'ctrl'),
          'xl':      (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          'xlnet':   (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          'xlm':     (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          'distil':  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          'roberta': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'xlm-roberta': (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
}

class VisnLangModel(nn.Module):
    """Language Model

    Args:
        param dim: dimension of the output
        param arch: backbone architecture
        param layers: final layer taken
        param weight_path: load pretrained weight
        param pretrained: load feature with pre-trained vector
        param finetuning: finetune the model
    """
    def __init__(self, dim = 512, arch='BERT', weight_path = None, layers=(-1,), pretrained=True, finetuning=True):
        super().__init__()
        Model, Tokenizer, weight = LANG_MODELS[arch]

        self.finetuning = finetuning
        self.pretrained = pretrained

        if weight_path is None:
            ### Load from
            bert = Model.from_pretrained(
                weight,
                output_hidden_states=True
            )
        else:
            bert = Model.from_pretrained(
                weight_path,
                output_hidden_states=True
            )

        if not pretrained:
            bert.init_weights()

        if not self.finetuning:
            for param in bert.parameters():
                param.requires_grad = False
        backbone_dim = bert.config.hidden_size
        self.backbone = bert
        self.layers = sorted(layers)

        print(f"Language Model: {arch} with weight {weight}; Fine-tuning: {finetuning}, Pre-trained: {pretrained}.")
        print(f"Language Model: using layers {self.layers}, result in backbone dim {backbone_dim * len(self.layers)} "
              f"--> output dim {dim}.")

        # Setup follow-up layers
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim * len(self.layers), 512 * len(self.layers)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512 * len(self.layers), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, dim)
        )

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        if not self.finetuning:
            with torch.no_grad():
                bert_output = self.backbone(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
        else:
            bert_output = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # sequence_output, pooled_output, (hidden_states), (attentions) --> seq_output
        if type(self.backbone) is XLNetModel:
            output, hidden_states = bert_output[:2]
        else:
            output, pooled_output, hidden_states = bert_output[:3]

        # gather the layers
        if type(self.backbone) is XLNetModel:
            x = torch.cat(list(hidden_states[layer].permute(1, 0, 2) for layer in self.layers), -1)
        else:
            x = torch.cat(list(hidden_states[layer] for layer in self.layers), -1)

        if not self.finetuning:
            x = x.detach()

        # [batch_size, max_len, backbone_dim] -->
        # [batch_size, max_len, output_dim]
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)

        # [batch_size, max_len, backbone_dim] -->
        # [batch_size, max_len, backbone_dim + output_dim]
        lang_visn_embedding = torch.cat((output, x),-1)
        lang_visn_pooled_output = torch.cat((pooled_output, x[:,0]),-1)

        new_bert_output = list(bert_output)
        new_bert_output[0] = lang_visn_embedding
        new_bert_output[1] = lang_visn_pooled_output
        #print(bert_output.last_hidden_state.shape)
        #print(bert_output.pooler_output.shape)
        return tuple(new_bert_output)