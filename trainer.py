# time: 2021/4/29 0:17
# File: trainer.py
# Author: zhangshubo
# Mail: supozhang@126.com
import torch
from tqdm import tqdm

from metric import accuracy
from utils import sequence_padding


class Trainer:

    def __init__(self, model, criterion, optimizer, learning_rate, device="cpu"):
        self.model = model
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.device = device

    def train_step(self, train_data, valid_data, vocab_dict, label_dict):
        running_loss = 0.0
        for i, (batch_x, batch_y_true) in tqdm(enumerate(train_data)):
            batch_x = list(map(lambda x: sequence_padding(x, 64), map(vocab_dict.lookup, batch_x)))
            batch_x = torch.tensor(batch_x, dtype=torch.long).to(self.device)
            batch_y_true = list(map(lambda x: sequence_padding(x, 64), map(label_dict.lookup, map(lambda x: x.split(" "), batch_y_true))))
            batch_y_true = torch.tensor(batch_y_true, dtype=torch.long).to(self.device)
            out = self.model(batch_x)
            loss = self.criterion(out, batch_y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(running_loss/200)
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


class LabellingTrainer:

    def __init__(self, model, criterion, optimizer, learning_rate, device="cpu"):
        self.model = model
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.device = device

    def train_step(self, train_data, valid_data, vocab_dict, label_dict):
        running_loss = 0.0
        for i, (batch_x, batch_y_true) in tqdm(enumerate(train_data)):
            batch_x = list(map(lambda x: sequence_padding(x, 64), map(vocab_dict.lookup, batch_x)))
            mask = [[0 if not i else 1 for i in x] for x in batch_x]
            batch_x = torch.tensor(batch_x, dtype=torch.long).to(self.device)
            # mask = torch.tensor(mask, dtype=torch.long).to(self.device)
            batch_y_true = list(map(lambda x: sequence_padding(x, 64), map(label_dict.lookup, map(lambda x: x.split(" "), batch_y_true))))
            batch_y_true = torch.tensor(batch_y_true, dtype=torch.long).to(self.device)
            out = self.model(batch_x)
            # loss = self.criterion(out.view(-1, self.mode.output_size) * mask.view(-1).unsqueeze(-1),
            #                       batch_y_true.view(-1)*mask.view(-1))
            loss = self.criterion(self.model.crf, batch_y_true, out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i+1) % 25 == 0:
                print(running_loss/25)
                running_loss = 0.0
        self.validation(valid_data, vocab_dict, label_dict)

    def validation(self, valid_data, vocab_dict, label_dict):
        all_y_true = []
        all_y_predict = []
        for i, (batch_x, batch_y_true) in tqdm(enumerate(valid_data)):
            batch_x = list(map(lambda x: sequence_padding(x, 64), map(vocab_dict.lookup, batch_x)))
            batch_x = torch.tensor(batch_x, dtype=torch.long).to(self.device)
            batch_y_true = list(map(lambda x: sequence_padding(x, 64),
                                    map(label_dict.lookup, map(lambda x: x.split(" "), batch_y_true))))
            out = self.model(batch_x)
            if self.model.learn_mode == "join":
                out = self.model.crf.viterbi_decoding(out)
            else:
                out = torch.argmax(out, dim=2)
            all_y_predict.extend(out.contiguous().view(-1).cpu().numpy())
            all_y_true.extend(torch.tensor(batch_y_true, dtype=torch.long).view(-1).numpy())
        print(f"validation accuracy: {accuracy(all_y_true, all_y_predict)}")

    def save_model(self, postfix="0"):
        torch.save(self.model, f"model_{postfix}.bin")