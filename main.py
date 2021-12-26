from kNN import createDataSet
import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures

def figure_show(array):
    assert array.shape[1] == 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array[:, 0], array[:, 1])
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
    mean_value = arr_random.mean(0)
    min_value = arr_random.min(0)
    max_value = arr_random.max(0)
    result_normalization = (arr_random - mean_value) / (max_value - min_value)
    return result_normalization


if __name__ == '__main__':
    arr_random = np.random.randn(100, 2)
    arr_rescaling = rescaling(arr_random)
    arr_normalization = normalization(arr_random)

    with futures.ProcessPoolExecutor(4) as executor:
        executor.submit(figure_show,arr_random)
        executor.submit(figure_show, arr_rescaling)
        executor.submit(figure_show, arr_normalization)
    pass
