import torch
import os
from pytorch_pretrained_bert import BertModel, BertTokenizer


class BertConfig(object):
    """
    配置参数
    """
    def __init__(self, bert_pretrain_path):
        self.model_name = 'Bert'

        # # 模型训练结果
        self.save_path = os.path.join(r".", self.model_name + '.ckpt')

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = 8
        # epoch数
        self.num_epochs = 50
        # batch_size
        self.batch_size = 80
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 0.00001
        # bert预训练模型位置
        self.bert_path = "E:/Workspaces/Python/NLP/Bruce-Bert-Text-Classification/bert_pretrain"
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层层个数
        self.hidden_size = 768