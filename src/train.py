# -*- coding: utf-8 -*-
# @Time    : 2022/7/6 22:14
# @Author  : zxf
import os
import json

import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel

from common.common import logger
from utils.util import REDataset
from utils.util import read_data
from utils.util import collate_fn
from utils.util import get_tokenizer
from models.rel_model import RelModel
from utils.util import read_dict_file
from utils.util import model_evaluate
from config.getConfig import get_config
from utils.util import set_global_random_seed


def train(config_init):
    Config = get_config(config_init)
    os.environ["CUDA_VISIBLE_DEVICES"] = Config["visible_gpus"]

    rank_num = len(Config["visible_gpus"].split(','))
    if rank_num > 1:
        dist.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = "cpu" if Config["visible_gpus"] == "-1" else "cuda"
    # check path
    if not os.path.exists(Config["output_path"]):
        os.mkdir(Config["output_path"])

    # set seed
    set_global_random_seed(Config["seed"])
    # read data
    train_data = read_data(Config["train_data_path"])
    dev_data = read_data(Config["dev_data_path"])
    logger.info("train data size:{}".format(len(train_data)))
    logger.info("dev data size:{}".format(len(dev_data)))

    # read rel dict
    rel2id_dict = read_dict_file(Config["rel_file"])
    rel2id = rel2id_dict[1]
    id2rel = rel2id_dict[0]
    # load tag dict
    tag2id_dict = read_dict_file(Config["tag_file"])
    tag2id = tag2id_dict[1]
    id2tag = tag2id_dict[0]
    logger.info("rel2id size:{}".format(len(rel2id)))
    logger.info("tag2id size:{}".format(len(tag2id)))

    # get tokenizer
    tokenizer = get_tokenizer(Config["pretrain_model_path"],
                              Config["pretrain_model_type"])

    # train dataset
    train_dataset = REDataset(train_data, rel2id, tag2id, Config["is_test"],
                              tokenizer, Config["max_seq_length"],
                              Config["replace_char"])
    dev_dataset = REDataset(dev_data, rel2id, tag2id, Config["is_test"],
                            tokenizer, Config["max_seq_length"],
                            Config["replace_char"])
    logger.info("train data size:{}".format(len(train_dataset)))
    logger.info("dev data size:{}".format(len(dev_dataset)))

    # dataloader
    if rank_num > 1:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=Config["batch_size"],
                                      shuffle=False,
                                      collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=False,
                                      batch_size=Config["batch_size"],
                                      collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    logger.info("pre epoch batch num:{}".format(len(train_dataloader)))
    logger.info("pre epoch batch num:{}".format(len(dev_dataloader)))

    # 如果需要下载预训练模型，这里在主程序进行下载
    if rank_num > 1:
        if local_rank not in [-1, 0]:
            dist.barrier()
    # onerel model
    model = RelModel(Config["pretrain_model_path"], Config["pretrain_model_type"],
                     len(rel2id), len(tag2id),
                     Config["dropout_prob"], Config["entity_pair_dropout"])
    model.to(device)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config["learning_rate"])

    # optimizer = AdamW(params=model.parameters(), lr=Config["learning_rate"])
    if rank_num > 1:
        if local_rank == 0:
            dist.barrier()

    if rank_num > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    best_f1 = 0
    loss_sum = 0

    model.train()
    for epoch in range(Config["epochs"]):
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["token_ids"].to(device)
            attention_mask = batch_data["mask"].to(device)
            loss_mask = batch_data["loss_mask"].to(device)
            triple_matrix = batch_data["triple_matrix"].to(device)
            # triples = batch_data["triples"]
            # tokens = batch_data["tokens"]
            _, loss = model(input_ids, attention_mask, triple_matrix,
                            loss_mask, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % Config["pre_epoch_step_print"] == 0:
                logger.info("epoch: {:3d}, step: {:4d}, train loss: {:5.3f}".
                             format(epoch+1, step+1, loss))

        if (epoch + 1) > Config["min_eval_epoch"]:
            logger.info("model evaluate")
            precision, recall, f1_score = model_evaluate(dev_dataloader, model, best_f1, id2rel, tag2id,
                   device, Config["output_evaluate_path"], Config["output_evaluate_file"],
                                                         Config["replace_char"])
            logger.info("epoch:{}, precision:{}, recall:{}, f1 :{}".format(epoch+1, precision,
                                                                           recall, f1_score))
            if f1_score >= best_f1:
                best_f1 = f1_score
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(Config["output_path"],
                                                                    Config["model_name"]))

        model.train()
    logger.info("best f1:{}".format(best_f1))


if __name__ == "__main__":
    config_init = "./config/config.ini"
    train(config_init)
