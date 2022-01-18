from collections import Counter

import numpy as np
from math import log


def get_entropy(list_label: list) -> float:
    """
    计算传入数组的最后一列的熵：sum(-P(x)logP(x) )
    :param list_label: list
    :return: 熵
    """
    count = len(list_label)
    property_conut = Counter(list_label)
    sum_value = 0
    for key, values in property_conut.items():
        pre_value = values / count
        log_value = log(pre_value, 2)
        sum_value = sum_value + pre_value * log_value
    sum_value = -sum_value
    return sum_value


def get_data():
    base_data = np.concatenate(
        (np.random.randint(0, 3, 1000).reshape(-1, 1), np.random.randint(0, 2, 1000).reshape(-1, 1)), axis=1)
    base_data = np.concatenate(
        (base_data, np.random.randint(0, 5, 1000).reshape(-1, 1)), axis=1)
    base_data = np.array([[1, 1],
                          [1, 1],
                          [1, 0],
                          [0, 1],
                          [0, 1]])
    label = ['yes', 'yes', 'no', 'no', 'no']
    return base_data, label


def rank_feature(dataset: np.ndarray, lable: list) -> dict:
    result = dict()
    x_length = dataset.shape[0]
    for col in range(dataset.shape[1]):
        new_entropy = 0
        property_count = Counter(dataset[:, col])
        for key, values in property_count.items():
            label_index = np.where(dataset[:, col] == key)[0].tolist()
            target_label = [lable[i] for i in label_index]
            # 获取筛选后数据集的信息熵
            property_entropy = get_entropy(target_label)
            new_entropy = new_entropy + (values / x_length) * property_entropy
        result[col] = new_entropy
    return result


if __name__ == '__main__':
    # 先 生成fake数据
    my_data, my_label = get_data()
    # 获取数据的熵
    entropy = get_entropy(my_label)
    # 返回按属性值划分后的数据集
    data_rank = rank_feature(my_data, my_label)
    # 选取值最小的列，作为最先划分的列。

    pass
