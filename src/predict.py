# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 13:59
# @Author  : zxf
import os
import json

import torch
import numpy as np

from common.common import logger
from utils.util import get_tokenizer
from utils.util import read_dict_file
from models.rel_model import RelModel


def create_data_infer_features(text, tokenizer, max_seq_len, replace_char, rel_num):
    text = replace_char.join(text.split())
    tokens = tokenizer.tokenize(text)
    # tokenizer 替换为transformers 中的tokenizer
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[: max_seq_len - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    text_len = len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    masks = [1] * len(token_ids)
    if len(token_ids) > text_len:
        token_ids = token_ids[:text_len]
        masks = masks[:text_len]
    return token_ids, masks, text_len, tokens


def predict(text, model_file, pretrain_model_path, pretrain_model_type, device,
            max_seq_len, rel_file, tag_file, dropout_prob, entity_pair_dropout, replace_char):
    # load relfile
    rel2id_dict = read_dict_file(rel_file)
    rel2id = rel2id_dict[1]
    id2rel = rel2id_dict[0]
    # load tag dict
    tag2id_dict = read_dict_file(tag_file)
    tag2id = tag2id_dict[1]
    id2tag = tag2id_dict[0]
    logger.info("rel2id size:{}".format(len(rel2id)))
    logger.info("tag2id size:{}".format(len(tag2id)))

    # get tokenizer
    tokenizer = get_tokenizer(pretrain_model_path, pretrain_model_type)

    # model
    model = RelModel(pretrain_model_path, pretrain_model_type,
                     len(rel2id), len(tag2id),
                     dropout_prob, entity_pair_dropout)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)),
                          strict=True)
    model.to(device)

    #  get text feature
    token_ids, attention_mask, text_len, tokens = \
        create_data_infer_features(text, tokenizer, max_seq_len, replace_char, len(rel2id))
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        pred_triple_matrix = model(input_ids, attention_mask, triple_matrix=None, loss_mask=None, train=False)
        pred_triple_matrix = pred_triple_matrix.data.cpu()[0]
        rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
        relations, heads, tails = np.where(pred_triple_matrix > 0)
        # 解码
        triple_list = []
        pair_numbers = len(relations)
        if pair_numbers > 0:
            # print('current sentence contains {} triple_pairs'.format(pair_numbers))
            for i in range(pair_numbers):
                r_index = relations[i]
                h_start_index = heads[i]
                t_start_index = tails[i]
                # 如果当前第一个标签为HB-TB
                if pred_triple_matrix[r_index][h_start_index][t_start_index] == tag2id[
                    'HB-TB'] and i + 1 < pair_numbers:
                    # 如果下一个标签为HB-TE
                    t_end_index = tails[i + 1]
                    if pred_triple_matrix[r_index][h_start_index][t_end_index] == tag2id['HB-TE']:
                        # 那么就向下找
                        for h_end_index in range(h_start_index, seq_lens):
                            # 向下找到了结尾位置
                            if pred_triple_matrix[r_index][h_end_index][t_end_index] == tag2id['HE-TE']:

                                sub_head, sub_tail = h_start_index, h_end_index
                                obj_head, obj_tail = t_start_index, t_end_index
                                sub = tokens[sub_head: sub_tail + 1]
                                # sub
                                sub = ''.join([i.lstrip("##") for i in sub])
                                obj = tokens[obj_head: obj_tail + 1]
                                # obj
                                obj = ''.join([i.lstrip("##") for i in obj])
                                rel = id2rel[str(int(r_index))]
                                if len(sub) > 0 and len(obj) > 0:
                                    triple_list.append((sub, rel, obj))
                                break
        triple_set = set()

        for s, r, o in triple_list:
            triple_set.add((s, r, o))

        pred_list = [list(item) for item in list(triple_set)]
    result = {"text": text,
              "label": pred_list}
    return result


if __name__ == "__main__":
    # text = "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下"
    text = "2019年2月25日和26日，温氏股份实控人之一、前任董事长温鹏程之妻伍翠珍分别减持公司股票608万股和256万股，成交均价分别为30.78元/股和30.02元/股，共计套现约2.64亿元"
    model_file = "./output/model.pt"
    # pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert-base-chinese/"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/"
    pretrain_model_type = "bert-base"
    device = "cpu"
    max_seq_len = 128
    rel_file = "./data/baidurelation2020/rel2id.json"
    tag_file = "./data/tag2id.json"
    dropout_prob = 0.2
    entity_pair_dropout = 0.1
    replace_char = '♠'
    result = predict(text, model_file, pretrain_model_path, pretrain_model_type, device,
                     max_seq_len, rel_file, tag_file, dropout_prob, entity_pair_dropout, replace_char)
    print("result: ", result)