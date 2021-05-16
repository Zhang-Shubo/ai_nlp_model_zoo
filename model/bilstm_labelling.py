# time: 2021/4/28 21:05
# File: bilstm_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch

from layer.crf import CRF
from layer.time_distributed import TimeDistributed


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_size, hidden_size,
                 learn_mode="join", test_mode="viterbi",
                 padding_idx=0, dropout_rate=0.1, device="cpu"):
        """
        Bi-LSTM mode for classifier
        :param vocab_size:
        :param output_size:
        :param embedding_size:
        :param hidden_size:
        :param padding_idx:
        :param dropout_rate:
        :param device:
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.learn_mode = learn_mode
        self.test_mode = test_mode

        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx).to(self.device)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size // 2,
            num_layers=1, bidirectional=True,
            batch_first=True
        ).to(self.device)
        self.dense = TimeDistributed(
            nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(self.hidden_size, self.output_size)).to(self.device),
            batch_first=True)
        self.crf = CRF(units=self.output_size, learn_mode=self.learn_mode, test_mode=self.test_mode, device=self.device)

    def forward(self, x_in):
        embedding = self.word_embedding(x_in)
        lstm_out, _ = self.lstm(embedding)
        out = self.dense(lstm_out)
        return out

