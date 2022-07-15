# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import AutoModel
from transformers import BertModel


class RelModel(nn.Module):
    def __init__(self, pretrain_model_path, pretrain_model_type, rel_num,
                 tag_size, dropout_prob, entity_pair_dropout):
        super(RelModel, self).__init__()
        self.rel_num = rel_num
        self.tag_size = tag_size
        self.dropout_prob = dropout_prob
        self.entity_pair_dropout = entity_pair_dropout
        if 'bert' in pretrain_model_type:
            self.bert_encoder = BertModel.from_pretrained(pretrain_model_path)
        else:
            self.bert_encoder = AutoModel.from_pretrained(pretrain_model_path)
        self.bert_dim = self.bert_encoder.config.hidden_size
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.rel_num * self.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.dropout_2 = nn.Dropout(self.entity_pair_dropout)
        self.activation = nn.ReLU()
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def get_encoded_text(self, input_ids, attention_mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return encoded_text

    def cal_loss(self, target, predict, mask):
        loss = self.loss_function(predict, target)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def triple_score_matrix(self, encoded_text, train=True):
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3
        batch_size, seq_len, bert_dim = encoded_text.size()
        # head: [batch_size, seq_len * seq_len, bert_dim(768)] 1,1,1, 2,2,2, 3,3,3
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)] 1,2,3, 1,2,3, 1,2,3
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len,
                                                                   self.rel_num, self.tag_size)
        if train:
            # [batch_size, tag_size, rel_num, seq_len, seq_len]
            return triple_scores.permute(0, 4, 3, 1, 2)
        else:
            # [batch_size, seq_len, seq_len, rel_num]
            return triple_scores.argmax(dim=-1).permute(0, 3, 1, 2)

    def forward(self, input_ids, attention_mask, triple_matrix=None, loss_mask=None,
                train=True):
        # [batch_size, seq_len]
        # token_ids = data['token_ids']
        # [batch_size, seq_len]
        # mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(input_ids, attention_mask)
        encoded_text = self.dropout(encoded_text)
        # [batch_size, rel_num, seq_len, seq_len]
        output = self.triple_score_matrix(encoded_text, train)
        if train:
            loss = self.cal_loss(triple_matrix, output, loss_mask)
            return output, loss

        return output
