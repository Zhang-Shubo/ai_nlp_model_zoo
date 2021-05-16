# time: 2021/4/29 1:45
# File: metric.py
# Author: zhangshubo
# Mail: supozhang@126.com
from collections import defaultdict


def accuracy(y_predict, y_true):
    length = len(y_predict)
    correct = 0
    for i, j in zip(y_predict, y_true):
        if i == j:
            correct += 1
    return correct / length


def parse_ner(tokens):
    """
    解析ner结果
    """
    res = {}
    began = False
    start = -1

    current_ner = ""
    for i, token in enumerate(tokens):

        if token.endswith("_M") and began:
            if current_ner == token[:-2]:
                continue

        if began and start != i - 1:
            res.setdefault(current_ner.lower(), []).append((start, i - 1))

        if token.endswith("_B"):
            began = True
            start = i
            current_ner = token[:-2]
            continue

        if token.endswith("_S"):
            res.setdefault(token[:-2].lower(), []).append([i, i])

        began = False
        start = -1
    return res


def ner_f1(y_true, y_predict):
    y_predict_ner = []
    y_true_ner = []
    true_dict = defaultdict(int)
    true_positive_dict = defaultdict(int)
    positive_dict = defaultdict(int)

    for predict in y_predict:
        y_predict_ner.append(parse_ner(predict))

    for true in y_true:
        y_true_ner.append(parse_ner(true))

    for true, predict in zip(y_true_ner, y_predict_ner):
        for ne, values in true.items():
            true_dict[ne] += len(values)

            if ne in predict:
                for val in values:
                    if val in predict[ne]:
                        true_positive_dict[ne] += 1
        for ne, values in predict.items():
            positive_dict[ne] += len(values)

    precise = sum(true_positive_dict.values()) / sum(positive_dict.values())
    recall = sum(true_positive_dict.values()) / sum(true_dict.values())
    f1 = 2*precise*recall / (precise+recall)
    return f1
