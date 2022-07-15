# -*- coding: utf-8 -*-
# @Time    : 2022/7/3 11:10
# @Author  : zxf
import os
import json
import time
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer


def get_tokenizer(pretrain_model_path, pretrain_model_type):
    """加载tokenizer"""
    if 'bert' in pretrain_model_type:
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    return tokenizer


def read_data(data_file):
    """读取json数据"""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_dict_file(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class REDataset(Dataset):
    def __init__(self, data, rel2id, tag2id, is_test,
                 tokenizer, max_seq_len, replace_char):
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.replace_char = replace_char

        self.data = data
        # self.rel2id = json.load(rel2id_fileopen(os.path.join(self.config.data_path, 'rel2id.json'), "r",
        #                              encoding="utf-8"))[1]
        self.rel2id = rel2id
        self.tag2id = tag2id
        self.rel_num = len(self.rel2id)
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        # self.cls_id = self.tokenizer.cls_token_id
        # self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json_data = self.data[idx]
        text = ins_json_data['text']
        # 将空格替换为一个特殊符号☉♠
        text = self.replace_char.join(text.split())
        # todo 这里的tokenizer 修改为了transformers
        tokens = self.tokenizer.tokenize(text)
        # tokenizer 替换为transformers 中的tokenizer
        if len(tokens) >= self.max_seq_len - 1:
            tokens = tokens[: self.max_seq_len - 2]
        tokens = [self.cls_token] + tokens + [self.sep_token]
        text_len = len(tokens)

        if not self.is_test:
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                # triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_tokens = self.tokenizer.tokenize(self.replace_char.join(triple[0].split()))
                obj_tokens = self.tokenizer.tokenize(self.replace_char.join(triple[2].split()))

                sub_head_idx = find_head_idx(tokens, sub_tokens)
                obj_head_idx = find_head_idx(tokens, obj_tokens)
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(sub_tokens) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(obj_tokens) - 1,
                                          self.rel2id[triple[1]]))

            # token_ids, segment_ids = self.tokenizer.encode(first=text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            masks = segment_ids
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            mask_length = len(masks)
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            loss_masks = np.ones((mask_length, mask_length))
            triple_matrix = np.zeros((self.rel_num, text_len, text_len))
            for s in s2ro_map:
                sub_head = s[0]
                sub_tail = s[1]
                for ro in s2ro_map.get((sub_head, sub_tail), []):
                    obj_head, obj_tail, relation = ro
                    triple_matrix[relation][sub_head][obj_head] = self.tag2id['HB-TB']
                    triple_matrix[relation][sub_head][obj_tail] = self.tag2id['HB-TE']
                    triple_matrix[relation][sub_tail][obj_tail] = self.tag2id['HE-TE']

            return token_ids, masks, loss_masks, text_len, triple_matrix, \
                   ins_json_data['triple_list'], tokens, self.rel_num

        else:
            # token_ids, masks = self.tokenizer.encode(first=text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(token_ids)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            mask_length = len(masks)
            loss_masks = np.array(masks) + 1
            triple_matrix = np.zeros((self.rel_num, text_len, text_len))
            return token_ids, masks, loss_masks, text_len, triple_matrix, \
                   ins_json_data['triple_list'], tokens, self.rel_num


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[3], reverse=True)

    token_ids, masks, loss_masks, text_len, triple_matrix, triples, tokens, rel_nums = zip(*batch)
    cur_batch_len = len(batch)
    max_text_len = max(text_len)
    rel_num = rel_nums[0]
    batch_token_ids = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_loss_masks = torch.LongTensor(cur_batch_len, 1, max_text_len, max_text_len).zero_()
    # if use WebNLG_star, modify 24 to 171, NTY为24
    batch_triple_matrix = torch.LongTensor(cur_batch_len, rel_num, max_text_len, max_text_len).zero_()

    for i in range(cur_batch_len):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_loss_masks[i, 0, :text_len[i], :text_len[i]].copy_(torch.from_numpy(loss_masks[i]))
        batch_triple_matrix[i, :, :text_len[i], :text_len[i]].copy_(torch.from_numpy(triple_matrix[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'loss_mask': batch_loss_masks,
            'triple_matrix': batch_triple_matrix,
            'triples': triples,
            'tokens': tokens}


def model_evaluate(test_dataloader, model, current_f1, id2rel, tag2id,
                   device, output_path,
                   output_file, replace_char, output=True):

    orders = ['subject', 'relation', 'object']

    def to_tup(triple_list):
        ret = []
        for triple in triple_list:
            ret.append(tuple(triple))
        return ret

    correct_num, predict_num, gold_num = 0, 0, 0

    results = []
    test_num = 0

    # while data is not None:
    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(test_dataloader):
            tokens = batch_data['tokens'][0]
            input_ids = batch_data["token_ids"].to(device)
            attention_mask = batch_data["mask"].to(device)
            loss_mask = batch_data["loss_mask"].to(device)
            triple_matrix = batch_data["triple_matrix"].to(device)
            # pred_triple_matrix: [1, rel_num, seq_len, seq_len]
            pred_triple_matrix = model(input_ids, attention_mask, triple_matrix,
                                       loss_mask, train=False)   # .cpu()[0]
            pred_triple_matrix = pred_triple_matrix.data.cpu()[0]
            rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
            relations, heads, tails = np.where(pred_triple_matrix > 0)

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
                                    # sub = ' '.join(sub.split('[unused1]')).strip()
                                    obj = tokens[obj_head: obj_tail + 1]
                                    # obj
                                    obj = ''.join([i.lstrip("##") for i in obj])
                                    # obj = ' '.join(obj.split('[unused1]')).strip()
                                    rel = id2rel[str(int(r_index))]
                                    if len(sub) > 0 and len(obj) > 0:
                                        triple_list.append((sub, rel, obj))
                                    break
            triple_set = set()

            for s, r, o in triple_list:
                triple_set.add((s, r, o))

            pred_list = list(triple_set)

            pred_triples = set(pred_list)
            gold_triples = set(to_tup(batch_data['triples'][0]))

            correct_num += len(pred_triples & gold_triples)
            predict_num += len(pred_triples)
            gold_num += len(gold_triples)

            if output:
                results.append({
                    'text': ''.join(tokens[1:-1]).replace(replace_char, ' ').replace('##', ''),
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                    ]
                })
        test_num += 1

    # print("\n correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    if output and f1_score > current_f1:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        path = os.path.join(output_path, output_file)

        fw = open(path, 'w', encoding="utf-8")

        for line in results:
            fw.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
        fw.close()
    return precision, recall, f1_score


def set_global_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False