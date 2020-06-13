import time
import torch
import numpy as np
from model.BERT.config import BertConfig
from model.BERT import utils
from model.BERT import train_BERT
from model.BERT.Classification_Bert import Classification_Bert

if __name__ == '__main__':
    dataset = 'datasets'  # 数据集地址
    config = BertConfig(dataset)

    ### 数据
    train_data = utils.bulid_dataset(
        dataset_path="E:/Workspaces/Python/KG/QA_healty39/data",
        tokenizer=config.tokenizer,
        pad_size=10)

    test_data = utils.bulid_test_dataset(
        dataset_path="E:/Workspaces/Python/KG/QA_healty39/data",
        tokenizer=config.tokenizer,
        pad_size=10
    )
    train_iter = utils.bulid_iterator(train_data, batch_size=config.batch_size, device=config.device)
    test_iter = utils.bulid_iterator(test_data, batch_size=config.batch_size, device=config.device)

    # 模型训练，评估与测试
    model = Classification_Bert()
    model.to(config.device)
    train_BERT.train(config, model, train_iter)
    acc, loss = train_BERT.evaluate(config, model, test_iter)
    print(acc, loss)
    # train_BERT.predict(model, config.tokenizer, ["#怎么治疗"])
