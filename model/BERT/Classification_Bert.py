import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Classification_Bert(nn.Module):
    """
    文本分类
    """
    def __init__(
        self,
        output_dim=8,           # 输出维度，分类类别数，8分类
        bert_hidden_dim=768,    # bert 的隐藏层个数
        n_layers=0,             # 后接的lstm层数
        lstm_hidden_dim=128,    # lstm的隐藏层个数
        dropout=0.3,
        bidirectional=True,     # bi-LSTM
        bert_pretrain_path="E:/Workspaces/Python/KG/QA_healty39/data/bert_pretrain"  # config中的参数
    ):
        super().__init__()  # 调用父类构造函数初始化从父类继承的变量
        self.num_directions = 2 if bidirectional else 1
        self.n_layers = n_layers

        # 模型
        self.bert = BertModel.from_pretrained(bert_pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(p=dropout)

        if n_layers != 0:
            self.lstm = nn.LSTM(
                input_size=bert_hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=0.1,
                batch_first=True
            )
            self.linear = nn.Linear(lstm_hidden_dim, output_dim)
        else:
            self.linear = nn.Linear(bert_hidden_dim, output_dim)

    def forward(self, x):
        # x [ids, seq_len, mask]
        # 输入的x： (ids, seq_len, mask) ==>  list(list(int)), list(int), list(list(int))
        context = x[0]  # 对应输入的句子 shape[batch_size, pad_size] (32, 32)
        mask = x[2]  # 对padding部分进行mask shape[128,32]

        # pooled 层每次那一句话的第一个单词输出作为dense层的输入， 因为self-attention不分先后
        # pooled: [batch_size, hidden_size]
        # encoded_layers [batch_size, sequence_length, hidden_size]
        encoded_layers, pooled = self.bert(context, attention_mask=mask,
                                           output_all_encoded_layers=False)  # shape [128,768]
        encoded_layers = self.dropout(encoded_layers)
        pooled = self.dropout(pooled)
        # hn(num_layers * num_directions, batch, hidden_size)

        if self.n_layers != 0:
            # 方式1
            output, (hn, cn) = self.lstm(encoded_layers)
            out = self.linear(hn[-1, :, :])  # shape [batchsize,300]

            # 方式2
            # output, (hn, cn) = self.LSTM(pooled.contiguous().view(-1, 1, self.bertConfig.hidden_size))
            # out = self.linear(hn[-1, :, :])  # shape [batchsize,300]

            # 方法3
            # output, (hn, cn) = self.LSTM(encoded_layers)
            # hn_new = hn[-2:, :, :]
            # hn_new = hn_new.reshape(hn_new.shape[1], hn_new.shape[2] * 2)
            # out = self.linear(hn_new)  # shape [batch_size,300]
        else:
            # 直接用bert的输出，不用 LSTM
            out = self.linear(pooled)
        return out
