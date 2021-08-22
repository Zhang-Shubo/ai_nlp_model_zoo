# time: 2021/4/28 21:05
# File: bilstm_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch

from layer.attention import attention


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_size, hidden_size,
                 padding_idx=0, connect_mode="average", dropout_rate=0.1,
                 weights=None, device="cpu"):
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
        self.weights = weights

        self.device = device

        assert connect_mode in ["average", "last", "attention"], \
            "Connect mode only support ['average', 'last', 'attention']"
        self.connect_mode = connect_mode
        self.padding_idx = padding_idx

        self.word_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx, _weight=self.weights).to(self.device)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size // 2,
            num_layers=1, bidirectional=True,
            batch_first=True
        ).to(self.device)
        self.query = nn.Parameter(torch.Tensor(512, hidden_size)).to(device)
        nn.init.xavier_normal(self.query)
        self.average_pool = nn.AdaptiveAvgPool1d(1).to(self.device)
        self.dense = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_size, self.output_size)
        ).to(self.device)

    def forward(self, x_in):
        embedding = self.word_embedding(x_in)
        lstm_out, _ = self.lstm(embedding)
        mask = (x_in.cpu() != self.padding_idx).int().to(self.device)
        num = torch.sum(mask, -1).squeeze(-1)
        if self.connect_mode == "average":
            linear_input = torch.transpose(lstm_out, 1, 2)
            linear_input = self.average_pool(linear_input).squeeze(-1) * torch.tensor(64/num).unsqueeze(-1)

        elif self.connect_mode == "attention":
            linear_input = attention(lstm_out[:, -1, :], lstm_out, lstm_out, mask)
            linear_input = torch.transpose(linear_input, 1, 2)
            linear_input = self.average_pool(linear_input).squeeze(-1) * torch.tensor(64/num).unsqueeze(-1)
        else:
            linear_input = lstm_out[:, -1, :]

        out = self.dense(linear_input)
        return out

