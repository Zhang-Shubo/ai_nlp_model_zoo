# time: 2021/4/28 23:17
# File: classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com
from tqdm import tqdm

from data_loader import ClassifierDataset
from torch.utils.data import DataLoader

from metric.metric import accuracy
from model.bert_classifier import BertClassifier
from model.cnn_classifier import TextCNN
from utils import VocabDict, LabelDict, sequence_padding, char_tokenizer

import torch


class Trainer:

    def __init__(self, model, criterion, optimizer, learning_rate, tokenizer=None, max_len=64, device="cpu"):
        self.model = model
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len

    def train_step(self, train_data, valid_data, vocab_dict, label_dict):
        running_loss = 0.0
        for i, (batch_x, batch_y_true) in tqdm(enumerate(train_data)):
            if callable(self.tokenizer):
                batch_x = self.tokenizer(batch_x, vocab_dict.lookup, self.max_len, self.device)
            else:
                batch_x = list(batch_x)
            batch_y_true = list(map(label_dict.lookup, batch_y_true))
            batch_y_true = torch.tensor(batch_y_true, dtype=torch.long).to(self.device)
            out = self.model(batch_x)
            loss = self.criterion(out, batch_y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(running_loss / 200)
                running_loss = 0.0
        self.validation(valid_data, vocab_dict, label_dict)

    def validation(self, valid_data, vocab_dict, label_dict):
        all_y_true = []
        all_y_predict = []
        for i, (batch_x, batch_y_true) in tqdm(enumerate(valid_data)):
            batch_x = list(map(lambda x: sequence_padding(x, 50, pos="pre"), map(vocab_dict.lookup, batch_x)))
            batch_x = torch.tensor(batch_x, dtype=torch.long).to(self.device)
            batch_y_true = list(map(label_dict.lookup, batch_y_true))
            out = self.model(batch_x)
            out = torch.argmax(out, dim=1)
            all_y_predict.extend(out.cpu().numpy())
            all_y_true.extend(batch_y_true)
        print(f"validation accuracy: {accuracy(all_y_true, all_y_predict)}")

    def save_model(self, postfix="0"):
        torch.save(self.model, f"model_{postfix}.bin")


def train():
    device = "cuda:0"
    train_dataset = ClassifierDataset("data/classifier/data/train.txt")
    valid_dataset = ClassifierDataset("data/classifier/data/dev.txt")
    train_data = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=256, shuffle=True)

    vocab_dict = VocabDict(train_dataset.get_all_inputs())
    label_dict = LabelDict(train_dataset.get_all_labels())

    # model = BiLSTM(len(vocab_dict), len(label_dict), embedding_size=300, hidden_size=512, device=device)
    # model = BertClassifier(len(label_dict), device=device)
    model = TextCNN(len(vocab_dict), len(label_dict), kernel_size_list=[1, 2, 3, 4, 5], filters=512, embedding_size=300,
                    device=device)
    trainer = Trainer(model, torch.nn.CrossEntropyLoss, torch.optim.Adam, tokenizer=char_tokenizer,
                      learning_rate=0.002, device=device)
    for i in range(30):
        trainer.train_step(train_data, valid_data, vocab_dict, label_dict)
        trainer.validation(valid_data, vocab_dict, label_dict)
    trainer.save_model(postfix="final")
    print(0)


if __name__ == '__main__':
    train()
