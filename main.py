import numpy as np
import matplotlib.pyplot as plt

def figure_show(array):
    assert array.shape[1] == 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array[:, 0], array[:, 1])
    plt.show()

if __name__ == '__main__':
    arr_random = np.random.randn(100, 2)
    figure_show(arr_random)

