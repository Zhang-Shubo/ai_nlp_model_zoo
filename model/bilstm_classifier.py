# time: 2021/4/28 21:05
# File: bilstm_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_size, hidden_size,
                 padding_idx=0, connect_mode="average", dropout_rate=0.1,
                 device="cpu"):
        """
        Bi-LSTM mode for classifier
        :param vocab_size:
        :param output_size:
        :param embedding_size:
        :param hidden_size:
        :param padding_idx:
        :param connect_mode:
        :param dropout_rate:
        :param device
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.device = device

        assert connect_mode in ["average", "last"], "Connect mode only support ['average', 'last']"
        self.connect_mode = connect_mode

        self.word_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx).to(self.device)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size // 2,
            num_layers=1, bidirectional=True,
            batch_first=True
        ).to(self.device)
        self.average_pool = nn.AdaptiveAvgPool1d(1).to(self.device)
        self.dense = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_size, self.output_size)
        ).to(self.device)

    def forward(self, x_in):
        embedding = self.word_embedding(x_in)
        lstm_out, _ = self.lstm(embedding)
        if self.connect_mode == "average":
            linear_input = torch.transpose(lstm_out, 1, 2)
            linear_input = self.average_pool(linear_input).squeeze(-1)
        else:
            linear_input = lstm_out[:, -1, :]

        out = self.dense(linear_input)
        return out

