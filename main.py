from kNN import createDataSet
import numpy as np
import matplotlib.pyplot as plt

def figure_show(array):
    assert array.shape[1]==2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(arr_random[:, 0], arr_random[:, 1])
    plt.show()


if __name__ == '__main__':
    arr_random = np.random.randn(200, 2)
    figure_show(arr_random)
    # 准备对arr_random 归一化处理： 取最大值、最小值，对每个值做运算；


