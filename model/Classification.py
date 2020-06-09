import torch
import torch.nn as nn


class Classification(nn.Module):
    """
    多层 lstm + linear
    """
    def __init__(
        self,
        vocab_size,         # 词表的长度（总单词数）， （词向量的数量）
        output_dim,         # 分类类别数
        n_layers=2,         # lstm 层数
        pad_idx=1,          # <unk>=0  <pad>=1
        hidden_dim=128,     # lstm 的hidden数量
        embed_dim=300,      # 300维词向量
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        num_directions = 1 if not bidirectional else 2
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )
        # (300， 128)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        # (128 * 2 * 1)
        self.linear = nn.Linear(hidden_dim * n_layers * num_directions, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        hidden_states, (h_n, c_c) = self.lstm(x)

        # 交换前两个shape，这样就变成了batchfirst格式的数据了， 并返回一个保存在整块内存中的新变量
        h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous()
        # h_n:(batch_size, hidden_size * num_layers * num_directions)
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.linear(h_n)
        return loggits
