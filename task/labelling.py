# time: 2021/5/12 23:17
# File: classifier.py
# Author: zhangshubo
# Mail: supozhang@126.com

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import LabellingDataset
from layer.crf import CRFLoss
from metric.metric import accuracy, ner_f1
from model.bert_labelling import BertLabelling
from model.bilstm_labelling import BiLSTM
from utils import VocabDict, LabelDict, sequence_padding, char_tokenizer, bert_tokenizer


class LabellingTrainer:

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
            batch_x_ = list(map(lambda x: sequence_padding(x, self.max_len), map(vocab_dict.lookup, batch_x)))
            mask = [[0 if not i else 1 for i in x] for x in batch_x_]
            mask = torch.tensor(mask, dtype=torch.long).to(self.device)

            batch_x = self.tokenizer(batch_x, vocab_dict.lookup, self.max_len, self.device)

            out = self.model(batch_x)
            batch_y_true = list(map(lambda x: sequence_padding(x, self.max_len),
                                    map(label_dict.lookup, map(lambda x: x.split(" "), batch_y_true))))
            batch_y_true = torch.tensor(batch_y_true, dtype=torch.long).to(self.device)

            # loss = self.criterion(out.view(-1, self.mode.output_size) * mask.view(-1).unsqueeze(-1),
            #                       batch_y_true.view(-1)*mask.view(-1))
            loss = self.criterion(self.model.crf, batch_y_true, out, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 25 == 0:
                print(running_loss / 25)
                running_loss = 0.0
        self.validation(valid_data, vocab_dict, label_dict)

    def validation(self, valid_data, vocab_dict, label_dict):
        all_y_true_idx = []
        all_y_predict_idx = []
        all_y_true_token = []
        all_y_predict_token = []
        for i, (batch_x, batch_y_true) in tqdm(enumerate(valid_data)):

            batch_x = self.tokenizer(batch_x, vocab_dict.lookup, self.max_len, self.device)

            batch_y_true_token = list(map(lambda x: x.split(" "), batch_y_true))
            batch_y_true = list(map(lambda x: sequence_padding(x, self.max_len),
                                    map(label_dict.lookup, batch_y_true_token)))
            out = self.model(batch_x)
            if self.model.learn_mode == "join":
                out = self.model.crf.viterbi_decoding(out)
            else:
                out = torch.argmax(out, dim=2)
            all_y_predict_idx.extend(out.contiguous().view(-1).cpu().numpy())
            all_y_true_idx.extend(torch.tensor(batch_y_true, dtype=torch.long).view(-1).numpy())
            all_y_true_token.extend(batch_y_true_token)
            all_y_predict_token.extend(list(map(label_dict.refactor, out.cpu().numpy().tolist())))
        print(f"validation token accuracy: {accuracy(all_y_true_idx, all_y_predict_idx)}")
        print(f"validation ner f1: {ner_f1(all_y_true_token, all_y_predict_token)}")

    def save_model(self, postfix="0"):
        torch.save(self.model, f"model_{postfix}.bin")


class BertLabellingTrainer:

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
            input_ids = self.tokenizer(batch_x, self.max_len, self.device)
            mask = [[0 if not i else 1 for i in x] for x in input_ids]
            mask = torch.tensor(mask, dtype=torch.long).to(self.device)

            out = self.model({"input_ids": input_ids, "attention_mask": mask})
            batch_y_true = list(map(lambda x: sequence_padding(x, self.max_len),
                                    map(lambda x: label_dict.lookup(x, begin=True, end=True),
                                        map(lambda x: x.split(" "), batch_y_true))))
            batch_y_true = torch.tensor(batch_y_true, dtype=torch.long).to(self.device)

            # loss = self.criterion(out.view(-1, self.mode.output_size) * mask.view(-1).unsqueeze(-1),
            #                       batch_y_true.view(-1)*mask.view(-1))
            loss = self.criterion(self.model.crf, batch_y_true, out, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(running_loss / 25)
                running_loss = 0.0
        self.validation(valid_data, vocab_dict, label_dict)

    def validation(self, valid_data, vocab_dict, label_dict):
        all_y_true_idx = []
        all_y_predict_idx = []
        all_y_true_token = []
        all_y_predict_token = []
        for i, (batch_x, batch_y_true) in tqdm(enumerate(valid_data)):

            input_ids = self.tokenizer(batch_x, self.max_len, self.device)
            mask = [[0 if not i else 1 for i in x] for x in input_ids]
            mask = torch.tensor(mask, dtype=torch.long).to(self.device)

            out = self.model({"input_ids": input_ids, "attention_mask": mask})
            batch_y_true_token = list(map(lambda x: ["O"] + x.split(" ") + ["O"], batch_y_true))
            batch_y_true = list(map(lambda x: sequence_padding(x, self.max_len),
                                    map(lambda x: label_dict.lookup(x, begin=True, end=True), batch_y_true_token)))

            if self.model.learn_mode == "join":
                out = self.model.crf.viterbi_decoding(out)
            else:
                out = torch.argmax(out, dim=2)
            all_y_predict_idx.extend(out.contiguous().view(-1).cpu().numpy())
            all_y_true_idx.extend(torch.tensor(batch_y_true, dtype=torch.long).view(-1).numpy())
            all_y_true_token.extend(batch_y_true_token)
            all_y_predict_token.extend(list(map(label_dict.refactor, out.cpu().numpy().tolist())))
        print(f"validation token accuracy: {accuracy(all_y_true_idx, all_y_predict_idx)}")
        print(f"validation ner f1: {ner_f1(all_y_true_token, all_y_predict_token)}")

    def save_model(self, postfix="0"):
        torch.save(self.model, f"model_{postfix}.bin")


def train():
    device = "cuda:0"
    train_dataset = LabellingDataset("data/labelling/data/train.json")
    valid_dataset = LabellingDataset("data/labelling/data/dev.json")
    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    vocab_dict = VocabDict(train_dataset.get_all_inputs())
    tag_dict = LabelDict(train_dataset.get_all_labels(), sequence=True)

    model = BiLSTM(len(vocab_dict), len(tag_dict), embedding_size=300, hidden_size=512, learn_mode="join",
                   device=device)
    trainer = LabellingTrainer(
        model, CRFLoss, torch.optim.Adam, tokenizer=char_tokenizer, learning_rate=0.002, device=device)
    # max_len = 64
    # model = BertLabelling(len(tag_dict), linear_dropout_rate=0.2, learn_mode="join", max_len=max_len,
    #                       device=device)
    # trainer = BertLabellingTrainer(model, CRFLoss, torch.optim.Adam, learning_rate=0.00003,
    #                                tokenizer=bert_tokenizer, max_len=max_len, device=device)

    for i in range(40):
        trainer.train_step(train_data, valid_data, vocab_dict, tag_dict)
    trainer.validation(valid_data, vocab_dict, tag_dict)
    trainer.save_model(postfix="final")
    print(0)


if __name__ == '__main__':
    train()
