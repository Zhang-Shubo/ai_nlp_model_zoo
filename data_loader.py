# time: 2021/4/28 22:55
# File: data_loader.py
# Author: zhangshubo
# Mail: supozhang@126.com

from torch.utils.data import Dataset

from utils import read_nlpcc_text, read_cluener_text


class ClassifierDataset(Dataset):

    def __init__(self, path):
        self.data = list(read_nlpcc_text(path))

    def __getitem__(self, item):
        text, label = self.data[item]
        return text, label

    def __len__(self):
        return len(self.data)

    def get_all_inputs(self):
        return [i[0] for i in self.data]

    def get_all_labels(self):
        return [i[1] for i in self.data]


class LabellingDataset(Dataset):

    def __init__(self, path):
        self.data = list(read_cluener_text(path))

    def __getitem__(self, item):
        text, label = self.data[item]
        return "".join(text), " ".join(label)

    def __len__(self):
        return len(self.data)

    def get_all_inputs(self):
        return [i[0] for i in self.data]

    def get_all_labels(self):
        return [i[1] for i in self.data]
