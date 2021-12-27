from collections import Counter

import numpy as np
import operator
import pandas as pd
from matplotlib import pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'C', 'D']
    return group, labels


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
    sorted_distances = distances.argsort()# 返回的是距离从小到大的索引值
    dict_result = Counter()
    for i in range(k):
        dict_result[y[sorted_distances[i]][0]] = dict_result[y[sorted_distances[i]][0]] + 1
    return dict_result


if __name__ == "__main__":
    data_url = "data/Ch02/datingTestSet.txt"
    arr_x, arr_y = load_data(data_url)
    count_y = Counter()
    dict_result = dict()
    for i in arr_y.tolist():
        if i[0] not in dict_result:
            dict_result.update({i[0]:len(dict_result)})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(arr_x[:, 1], arr_x[:, 2], 15.0, [dict_result[i] for i in arr_y[:, 0]])
    plt.show()


    target = [13438, 9.665618, 0.261333]
    result = classify0(target, 4, arr_x, arr_y)
