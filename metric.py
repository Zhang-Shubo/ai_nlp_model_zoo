# time: 2021/4/29 1:45
# File: metric.py
# Author: zhangshubo
# Mail: supozhang@126.com

def accuracy(y_predict, y_true):
    length = len(y_predict)
    correct = 0
    for i, j in zip(y_predict, y_true):
        if i == j:
            correct += 1
    return correct/length


def ner_f1(y_predict, y_true):
    pass
