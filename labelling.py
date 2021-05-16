# time: 2021/5/12 23:17
# File: classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com
from data_loader import LabellingDataset
from torch.utils.data import DataLoader

from layer.crf import CRFLoss
from model.bilstm_labelling import BiLSTM
from trainer import LabellingTrainer
from utils import VocabDict, LabelDict

import torch

device = torch.device("cuda:0")
train_dataset = LabellingDataset("data/labelling/data/train.json")
valid_dataset = LabellingDataset("data/labelling/data/dev.json")
train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_data = DataLoader(valid_dataset, batch_size=64, shuffle=True)

vocab_dict = VocabDict(train_dataset.get_all_inputs())
tag_dict = LabelDict(train_dataset.get_all_labels(), sequence=True)

model = BiLSTM(len(vocab_dict), len(tag_dict), embedding_size=300, hidden_size=512, learn_mode="join", device=device)

trainer = LabellingTrainer(model, CRFLoss,
                           torch.optim.Adam, learning_rate=0.002, device=device)

for i in range(40):
    trainer.train_step(train_data, valid_data, vocab_dict, tag_dict)
trainer.validation(valid_data, vocab_dict, tag_dict)
trainer.save_model(postfix="final")
print(0)