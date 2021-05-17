# time: 2021/5/17 21:05
# File: bert_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel

from layer.crf import CRF
from layer.time_distributed import TimeDistributed


class BertLabelling(nn.Module):

    def __init__(self, output_size, linear_dropout_rate=0.2,
                 max_len=64,
                 learn_mode="join", test_mode="viterbi", device="cpu"):
        super().__init__()
        self.output_size = output_size
        self.linear_dropout_rate = linear_dropout_rate

        self.learn_mode = learn_mode
        self.test_mode = test_mode

        self.max_len = max_len

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("data/bert/bert-base-chinese/")
        self.bert_model = AutoModel.from_pretrained("data/bert/bert-base-chinese/")
        self.bert_model.to(device)
        self.out_layer = TimeDistributed(nn.Sequential(nn.Dropout(linear_dropout_rate), nn.Linear(768, output_size)),
                                         batch_first=True)
        self.out_layer.to(device)

        self.crf = CRF(units=self.output_size, learn_mode=self.learn_mode, test_mode=self.test_mode, device=self.device)

    def forward(self, x_in):
        inputs = self.tokenizer(x_in, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        for key, value in inputs.data.items():
            inputs.data[key] = value.to(self.device)
        embedding = self.bert_model(**inputs)

        out = self.out_layer(embedding["last_hidden_state"])

        return out, inputs.data["attention_mask"]
