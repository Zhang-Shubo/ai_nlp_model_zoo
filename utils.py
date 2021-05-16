# time: 2021/4/28 22:50
# File: utils.py
# Author: zhangshubo
# Mail: supozhang@126.com
import json
import os


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
        ner_list = ["O"]*len(word_list)
        for ner_type, value_dict in da["label"].items():
            for words, position_list in value_dict.items():
                for position in position_list:
                    if position[0] == position[1]:
                        ner_list[position[0]] = ner_type.upper()+"_S"
                    else:
                        ner_list[position[0]] = ner_type.upper()+"_B"
                        for pos in range(position[0]+1, position[1]+1):
                            ner_list[pos] = ner_type.upper()+"_M"
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

    def lookup(self, label):
        if self.is_sequence:
            res = []
            for tag in label:
                if tag in self:
                    res.append(self[tag])
            return res
        return self[label]


def sequence_padding(seq, max_len, pos="post", pad_idx=0):
    z = [pad_idx] * max_len
    if len(seq) > max_len:
        seq = seq[:max_len]
    if pos == "post":
        z[:len(seq)] = seq
    else:
        z[-len(seq):] = seq
    return z
