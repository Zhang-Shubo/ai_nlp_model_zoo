# time: 2021/5/17 21:05
# File: bert_classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel


class BertClassifier(nn.Module):

    def __init__(self, output_size, connect_mode="cls", linear_dropout_rate=0.2, device="cpu"):
        super().__init__()
        self.output_size = output_size
        self.connect_mode = connect_mode
        self.linear_dropout_rate = linear_dropout_rate
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("data/bert/bert-base-chinese/")
        self.bert_model = AutoModel.from_pretrained("data/bert/bert-base-chinese/")
        self.bert_model.to(device)
        self.out_layer = nn.Sequential(nn.Dropout(linear_dropout_rate), nn.Linear(768, output_size))
        self.out_layer.to(device)
        self.average_pool = nn.AdaptiveAvgPool1d(1).to(device)

    def forward(self, x_in):
        inputs = self.tokenizer(x_in, return_tensors="pt", padding=True, truncation=True, max_length=128)
        for key, value in inputs.data.items():
            inputs.data[key] = value.to(self.device)
        embedding = self.bert_model(**inputs)

        if self.connect_mode == "average":
            linear_input = torch.transpose(embedding["last_hidden_state"], 1, 2)
            linear_input = self.average_pool(linear_input).squeeze(-1)
        else:
            linear_input = embedding["pooler_output"]
        out = self.out_layer(linear_input)
        return out

