import numpy as np
import operator
import pandas as pd


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'C', 'D']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    分类器
    :param inX: 待分类的输入向量
    :param dataSet: 训练样本集
    :param labels: 训练样本标签
    :param k: 选取前k个临近的
    :return:
    """
    assert len(inX) == dataSet.shape[1]
    assert k <= dataSet.shape[0]
    dataSetSize = dataSet.shape[0]
    # 将inX复制dataSet集合的个数;np.tile(inX, )如果是一个int，就是将inX的属性复制int次
    # 如果是（x，y）就是将inX的属性复制y次，行复制x次,所以下面代码的意思是让inX的长度与dataSet一致
    align_inX = np.tile(inX, (dataSetSize, 1))
    # 分别做差
    diffMat = align_inX - dataSet
    # 各自平方
    sqDiffMat = diffMat ** 2
    # 将属性的平方和相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号
    distances = sqDistances ** 0.5
    # 返回的是距离从小到大的索引值
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


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
    data_url = "machinelearninginaction/Ch02/datingTestSet.txt"
    arr_x, arr_y = load_data(data_url)
    target = [13438,9.665618,0.261333]
    classify1(target, 1, arr_x, arr_y)
