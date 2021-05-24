# time: 2021/5/24 21:49
# File: neural_function.py
# Author: zhangshubo
# Mail: supozhang@126.com
import numpy as np


def softmax(x, axis=1):
    row_max = np.max(x, axis=axis, keepdims=True)
    x = x - row_max

    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_x


if __name__ == '__main__':
    a = np.random.random([3, 3])
    b = softmax(a)
    print(0)