# from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os
from gensim import corpora
# import numpy as np
import random

PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path="../datasets/", tokenizer=None, pad_size=10):
    """
    :param file_path:
    :param seq_len:
    :param pad_size:
    :return: list((list(id), int(label), int(len), list(mask)))
    """
    contents = []
    labels = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.split('\t')
            question, intention = line

            labels.append(intention)

            token = tokenizer.tokenize(question)    # 切成字列表，list(str)
            token = [CLS] + token
            seq_len = len(token)        # 真实的句子长度
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)   # 转换成对应的id列表， list(int)

            if pad_size:
                # 句子太短
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                # 句子太长
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size      # 切断句子的长度，最大32
            contents.append((token_ids, intention, seq_len, mask))

        # 针对id建立词典
        dictory = corpora.Dictionary([labels])
        # 将label 转换为id
        contents = [(token_ids, dictory.token2id[label], seq_len, mask) for (token_ids, label, seq_len, mask) in contents]
        random.shuffle(contents)
        print("### 打乱数据")
    return contents


def load_dataset_from_input(input, tokenizer=None, pad_size=10, device=None):
    """
    将输入数据转换为 [list(int), int, list(int)]
    :param input:
    :param tokenizer:
    :param pad_size:
    :return:
    """
    contents = []
    input = input.strip()
    token = tokenizer.tokenize(input)  # 切成字列表，list(str)
    token = [CLS] + token
    seq_len = len(token)  # 真实的句子长度
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)  # 转换成对应的id列表， list(int)

    if pad_size:
        # 句子太短
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids = token_ids + ([0] * (pad_size - len(token)))
        # 句子太长
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size  # 切断句子的长度，最大32

    token_ids = torch.LongTensor(token_ids).to(device)
    seq_len = torch.LongTensor(seq_len).to(device)
    mask = torch.LongTensor(mask).to(device)
    contents.append(([token_ids], seq_len, [mask]))
    return contents


# file_path="../datasets/", tokenizer=None, pad_size=32
def bulid_dataset(dataset_path="E:/Workspaces/Python/KG/QA_healty39/data", tokenizer=None, pad_size=10):
    datasetpkl = dataset_path + "/qa.pkl"
    if os.path.exists(datasetpkl):
        train = pkl.load(open(datasetpkl, 'rb'))
    else:
        train = load_dataset(dataset_path + "/qa.csv", tokenizer, pad_size)
        pkl.dump(train, open(datasetpkl, 'wb'))
    return train


def bulid_test_dataset(dataset_path, tokenizer=None, pad_size=10):
    datasetpkl = dataset_path + "/qa_test.pkl"
    if os.path.exists(datasetpkl):
        train = pkl.load(open(datasetpkl, 'rb'))
    else:
        train = load_dataset(dataset_path + "/qa_test.csv", tokenizer, pad_size)
        pkl.dump(train, open(datasetpkl, 'wb'))
    return train


class DatasetIterator(object):
    """
    迭代器对象
    返回4个张量 ： (x, seq_len, mask), y  ==> (list(list(int)), list(int), list(list(int)), int)
    - 实现了__iter__函数对象是 可迭代对象Iteratable，他应该返回一个实现了__next__的对象
    - 同时实现了__iter__和__next__的是迭代器(Iterator)
    - __next__函数中通过抛出一个StopIteration异常表示迭代结束
    """
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False            # 记录batch数量是否为整数

        if self.n_batches == 0 or len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)   # x样本数据 list(list(id))
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)   # 标签数据 int(label)

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)     # 每一个序列的真实长度 int(len)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)        # list(list(int))

        return (x, seq_len, mask), y

    def __next__(self):
        # 有余数， 并且已经到了最后一轮
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        # 结束
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        # 正常情况
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def bulid_iterator(dataset, batch_size, device):
    iter = DatasetIterator(dataset, batch_size, device)
    return iter


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))