#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 个人微信 wibrce
# Author 杨博
import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained_bert import BertAdam
from model.BERT import utils


def train(config, model, train_iter):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    # 启动 BatchNormalization 和 dropout
    model.train()
    # 拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0                 # 记录进行多少batch
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        # 迭代器返回的结果： (x, seq_len, mask), y  ==> (list(list(int)), list(int), list(list(int)), int)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 10 == 0:  # 每多少次输出在训练集和校验集上的效果
                true = labels.data.cpu()
                predit = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predit)
                print(f"##### {loss.item()}, acc {train_acc}")
                model.train()
            total_batch = total_batch + 1


def predict(model, tokenizer, input):
    input_vec = utils.load_dataset_from_input(input=input[0], tokenizer=tokenizer)
    output = model(input_vec[0])
    print(output)


def evaluate(config, model, dev_iter, test=False):
    """

    :param config:
    :param model:
    :param dev_iter:
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)
#
#
# def test(config, model, test_iter):
#     """
#     模型测试
#     :param config:
#     :param model:
#     :param test_iter:
#     :return:
#     """
#     # model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score")
#     print(test_report)
#     print("Confusion Maxtrix")
#     print(test_confusion)
#     time_dif = utils.get_time_dif(start_time)
#     print("使用时间：", time_dif)


















