# time: 2021/4/28 23:17
# File: classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com
from data_loader import ClassifierDataset
from torch.utils.data import DataLoader

from model.bilstm_classifier import BiLSTM
from trainer import Trainer
from utils import VocabDict, LabelDict

import torch

device = torch.device("cuda:0")
train_dataset = ClassifierDataset("data/classifier/data/train.txt")
valid_dataset = ClassifierDataset("data/classifier/data/dev.txt")
train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_data = DataLoader(valid_dataset, batch_size=64, shuffle=True)

vocab_dict = VocabDict(train_dataset.get_all_inputs())
label_dict = LabelDict(train_dataset.get_all_labels())

model = BiLSTM(len(vocab_dict), len(label_dict), embedding_size=300, hidden_size=512, device=device)

trainer = Trainer(model, torch.nn.CrossEntropyLoss, torch.optim.Adam, learning_rate=0.002, device=device)
for i in range(5):
    trainer.train_step(train_data, valid_data, vocab_dict, label_dict)
    trainer.validation(valid_data, vocab_dict, label_dict)
trainer.save_model(postfix="final")
print(0)