import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BERT_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.sequence_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        #self.token_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.label_ignore_id = config.label_ignore_id

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
            weights=None,
            token_cls=None,
            sequence_cls=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(bert_output.last_hidden_state)
        sequence_cls_logits = self.sequence_classifier(bert_output.pooler_output)
        #token_cls_logits = self.token_classifier(bert_output.last_hidden_state)
        loss = None
        if trg_ids is not None:
            cls_loss_function = nn.CrossEntropyLoss(weight = torch.Tensor([1, 6]).to('cuda'))#, ignore_index=-1)
            logsoftmax_func=nn.LogSoftmax(dim=1)
           
            resize_pred = logits.view(-1, self.bert.config.vocab_size)
            N, C = resize_pred.size(0), resize_pred.size(1)
            resize_label = trg_ids.view(N, 1)
            resize_weights = weights.view(1, N).float()
            class_mask = resize_pred.data.new(N, C).fill_(0)
            class_mask = class_mask.float().requires_grad_()
            class_mask.data.scatter_(1, resize_label.data, -1.)
            
            logsoftmax_output=logsoftmax_func(resize_pred)
            probs = (logsoftmax_output * class_mask).sum(1).view(-1, 1)
            loss = torch.mm(resize_weights, probs) / torch.count_nonzero(trg_ids).item()
            loss += cls_loss_function(sequence_cls_logits.view(-1, 2), sequence_cls.view(-1))
            #loss += cls_loss_function(token_cls_logits.view(-1, 2), token_cls.view(-1))
        return loss, logits
    
    def tie_cls_weight(self):
            self.classifier.weight = self.bert.embeddings.word_embeddings.weight
    
    
