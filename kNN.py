from collections import Counter

import numpy as np
import operator
import pandas as pd
from matplotlib import pyplot as plt


def show_data():
    data_url = "data/Ch02/datingTestSet.txt"
    arr_x, arr_y = load_data(data_url)
    target = [13438, 9.665618, 0.261333]
    result = classify0(target, 4, arr_x, arr_y)
    dict_result = dict()
    for i in arr_y.tolist():
        if i[0] not in dict_result:
            dict_result.update({i[0]: len(dict_result)})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(arr_x[:, 1], arr_x[:, 2], 15.0, [dict_result[i] for i in arr_y[:, 0]])
    plt.show()


def rescaling(arr_random):
    """
    数据归一化
    :param arr_random:
    :return:
    """
    min_value = arr_random.min(0)
    max_value = arr_random.max(0)
    result_rescaling = (arr_random - min_value) / (max_value - min_value)
    return result_rescaling


def normalization(target_arr):
    """
    数据标准化
    :param target_arr:
    :return:
    """
    mean_value = target_arr.mean(0)
    min_value = target_arr.min(0)
    max_value = target_arr.max(0)
    result_normalization = (target_arr - mean_value) / (max_value - min_value)
    return result_normalization


def load_data(url):
    base_data = pd.read_csv(url, delim_whitespace=True, header=None)
    arr_x = base_data.loc[:, 0:2].values
    arr_y = base_data.loc[:, 3].values
    arr_y = arr_y.reshape((arr_y.shape[0], 1))
    return arr_x, arr_y


def classify0(targe_vector, k, x, y):
    """
    k-近邻算法，目前只支持输入一个元素
    :param targe_vector:目标元素
    :param k: 取前k个近似的数据
    :param x: 训练集
    :param y: 训练标签
    :return: 前k个的标签 {"good":5, "fine":3}
    """
    assert len(targe_vector) == x.shape[1]
    # 平方差求和，再取根号
    distances = ((x - targe_vector) ** 2).sum(axis=1) ** 0.5
    sorted_distances = distances.argsort()  # 返回的是距离从小到大的索引值
    dict_result = Counter()
    for i in range(k):
        dict_result[y[sorted_distances[i]][0]] = dict_result[y[sorted_distances[i]][0]] + 1
    return dict(dict_result)


def datingClassTest():
    train_rate = 0.9
    data_url = "data/Ch02/datingTestSet.txt"
    arr_x, arr_y = load_data(data_url)
    arr_x = rescaling(arr_x)
    # 用于训练数据的数量
    train_count = int(arr_x.shape[0] * train_rate)
    test_x = arr_x[train_count:, :]
    test_y = arr_y[train_count:, :]
    dict_result = {"true": 0, "false": 0}
    for target_x, target_y in zip(test_x, test_y):
        pre_y = classify0(target_x, 1, arr_x[:train_count, :], arr_y[:train_count, :])
        if list(pre_y.keys())[0] == target_y:
            dict_result['true'] = dict_result['true'] + 1
        else:
            dict_result['false'] = dict_result['false'] + 1
    return dict_result


def load_data(url):
    base_data = pd.read_csv(url, delim_whitespace=True, header=None)
    arr_x = base_data.loc[:, 0:2].values
    arr_y = base_data.loc[:, 3].values
    arr_y = arr_y.reshape((arr_y.shape[0], 1))
    return arr_x, arr_y

def classify1(targe_vector, k, x, y):
    assert len(targe_vector)==x.shape[1]
    diff = x- targe_vector
    sq_diff = diff **2
    sq_distance = sq_diff.sum(axis=1)
    distances = sq_distance ** 0.5
    # distances = distances.reshape(-1, 1)
    sorted_distances = distances.argsort()
    voteIlabel = y[sorted_distances[1]]
    pass


if __name__ == "__main__":
    # 返回分类器的精度
    dict_result = datingClassTest()
    show_data()

