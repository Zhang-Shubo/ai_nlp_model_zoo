# time: 2021/4/28 22:50
# File: utils.py
# Author: zhangshubo
# Mail: supozhang@126.com
import json
import os
import random

import torch

_bert_token_dict = json.loads(open("data/bert/bert-base-chinese/tokenizer.json", encoding="utf-8").read())["model"][
    "vocab"]


def read_nlpcc_text(path):
    with open(path, "r", encoding="utf-8") as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            label, text = line.strip("\n").split("\t")
            yield text.replace(" ", ""), label


def read_cluener_text(path):
    def transfer(da):
        word_list = list(da["text"])
        ner_list = ["O"] * len(word_list)
        for ner_type, value_dict in da["label"].items():
            for words, position_list in value_dict.items():
                for position in position_list:
                    if position[0] == position[1]:
                        ner_list[position[0]] = ner_type.upper() + "_S"
                    else:
                        ner_list[position[0]] = ner_type.upper() + "_B"
                        for pos in range(position[0] + 1, position[1] + 1):
                            ner_list[pos] = ner_type.upper() + "_M"
        return word_list, ner_list

    with open(path, "r", encoding="utf-8") as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            data = json.loads(line)
            yield transfer(data)


class VocabDict(dict):
    def __init__(self, inputs, save_path="data/classifier/vocab.txt"):
        super(VocabDict, self).__init__({"<PAD>": 0, "<UNK>": 1})
        self.data = ["<PAD>", "<UNK>"]
        self.save_path = save_path
        self.weights = [[0.0] * 200, [random.random() for _ in range(200)]]
        if not self.load():
            self.traverse(inputs)

    def load(self):
        if not os.path.exists(self.save_path):
            return
        with open(self.save_path, encoding="utf-8") as fd:
            while True:
                line = fd.readline()
                if not line:
                    break
                self.data.append(line.strip())
        for i, word in enumerate(self.data):
            self[word] = i
        return True

    def load_pretrained_vocab(self, path):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fd:
            while True:
                line = fd.readline()
                if not line:
                    break
                blocks = line.strip().split(" ")
                self.data.append(blocks[0])
                self.weights.append(list(map(float, blocks[1:])))
        for i, word in enumerate(self.data):
            self[word] = i
        return self

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as fd:
            for word in self.data:
                fd.write(word + "\n")

    def traverse(self, inputs):
        for line in inputs:
            for w in line:
                if w not in self:
                    self[w] = len(self)
                    self.data.append(w)

    def lookup(self, seq):
        res = []
        for char in seq:
            if char in self:
                res.append(self[char])
            else:
                res.append(self["<UNK>"])
        return res


class LabelDict(dict):
    def __init__(self, labels, save_path="data/classifier/label.txt", sequence=False):
        super(LabelDict, self).__init__()
        self.data = []
        self.save_path = save_path
        self.is_sequence = sequence
        if self.is_sequence:
            self["O"] = 0
            self.data.append("O")
        if not self.load():
            self.traverse(labels)

    def load(self):
        if not os.path.exists(self.save_path):
            return
        with open(self.save_path, encoding="utf-8") as fd:
            while True:
                line = fd.readline()
                if not line:
                    break
                self.data.append(line.strip())
        for i, label in enumerate(self.data):
            self[label] = i
        return True

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as fd:
            for label in self.data:
                fd.write(label + "\n")

    def traverse(self, labels):
        for label in labels:
            if not self.is_sequence:
                if label not in self:
                    self[label] = len(self)
                    self.data.append(label)
            else:
                for tag in label:
                    if tag not in self:
                        self[tag] = len(self)
                        self.data.append(tag)

    def lookup(self, label, begin=False, end=False):
        """
        将标签转化为数字

        """
        if self.is_sequence:
            res = []
            if begin:
                res.append(self["O"])
            for tag in label:
                if tag in self:
                    res.append(self[tag])
            if end:
                res.append(self["O"])
            return res
        return self[label]

    def refactor(self, predict):
        """
        将数字转化为标签
        :param predict:
        :return:
        """
        if not self.is_sequence:
            return self.data[predict]
        else:
            res = []
            for tag_idx in predict:
                res.append(self.data[tag_idx])
            return res


def sequence_padding(seq, max_len, pos="post", pad_idx=0):
    z = [pad_idx] * max_len
    if len(seq) > max_len:
        seq = seq[:max_len]
    if pos == "post":
        z[:len(seq)] = seq
    else:
        z[-len(seq):] = seq
    return z


def char_tokenizer(batch_x, lookup_f, max_len, device, padding_pos="post"):
    batch_x = list(map(lambda x: sequence_padding(x, max_len, pos=padding_pos), map(lookup_f, batch_x)))
    batch_x = torch.tensor(batch_x, dtype=torch.long).to(device)
    return batch_x


def bert_tokenizer(batch_x, max_len, device):
    def lookup(x):
        res = [_bert_token_dict["[CLS]"]]
        for char in x:
            if char in _bert_token_dict:
                res.append(_bert_token_dict[char])
            else:
                res.append(_bert_token_dict["[UNK]"])
        res.append(_bert_token_dict["[SEP]"])
        return res

    batch_x = list(map(lambda x: sequence_padding(x, max_len), map(lookup, batch_x)))
    batch_x = torch.tensor(batch_x, dtype=torch.long).to(device)
    return batch_x


def extra_tencent_embedding(path):
    res = []
    with open(path, encoding="utf-8") as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            if len(line.strip().split(" ")[0]) == 1:
                res.append(line)
    with open("data/tencent_char_embedding.txt", "w", encoding="utf-8") as fd:
        for line in res:
            fd.write(line)


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

# extra_tencent_embedding(r"E:\tencent_embedding\Tencent_AILab_ChineseEmbedding.txt")
