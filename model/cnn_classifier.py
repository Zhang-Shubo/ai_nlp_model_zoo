# time: 2021/4/28 21:05
# File: bilstm_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch


class TextCNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_size,
                 kernel_size_list, filters,
                 padding_idx=0, dropout_rate=0.1,
                 device="cpu"):
        """
        CNN mode for text classifier
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.kernel_size_list = kernel_size_list

        self.device = device
        self.filters = filters

        self.word_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx).to(self.device)
        self.cnn = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embedding_size, filters, (i,)),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for i in self.kernel_size_list
        ]).to(self.device)
        self.dense = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(self.filters*len(self.kernel_size_list), self.output_size)
        ).to(self.device)

    def forward(self, x_in):
        embedding = self.word_embedding(x_in).transpose(2, 1)

        cnn_out = [model(embedding) for model in self.cnn]
        cnn_out = torch.cat(cnn_out, dim=1).squeeze(-1)

        out = self.dense(cnn_out)
        return out
