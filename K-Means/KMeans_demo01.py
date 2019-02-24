"""
function:KMeans算法实现
author:gsy
date:2019.2.24
"""

import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape   # 10×2
    centroids = np.zeros((k, n)) # 4×2
    for i in range(k):  # 遍历1-4
        index = int(np.random.uniform(0, m))   #随机从十行数据中获取索引作为质心
        centroids[i, :] = dataSet[index, :]
        # print("index:",index)
    # print("centroids",centroids)
    """
    [[-5.37 -3.36]
    [ 2.67  1.59]
    [ 2.67  1.59]
    [ 2.67  1.59]]
    """
    return centroids


# k均值聚类
"""
算法思想：
1.根据启发式原则，将所给的数据点初始化为k蔟，并为每一蔟初始化质心
  原则：簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大
  在KMeans算法中，质心为蔟内点的均值
2.计算所有数据点到每个质心的距离，将数据点分配到距离最近的质心所属的蔟中
  更新质心的值
3.重复步骤2，直至质心不发生变化为止
"""
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # print(type(clusterAssment))  <class 'numpy.matrix'>
    # print(clusterAssment[:, 0].A)    .A是获取当前第1列数据

    # 第1步 初始化centroids，随机获取4个数据点作为质心
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        """
        此处遍历为两层循环：
        最外层对每一个样本点进行遍历，
        内层循环：依次对每个样本点计算其到4个质心的距离，将其归属到距离最小的那一蔟
        
        所有样本点做了一次判别后，则更新质心
        """
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])   #选取指定行，所有列
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:  # clusterAssment包含两列数据，第一列为样本属于哪一蔟，10×2
                clusterChange = True   # 只要更新了样本所属的蔟，即clusterChange变为True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()


dataSet = loadDataSet("test.txt")
# 初始化：将数据点分为4类
k = 4
centroids, clusterAssment = KMeans(dataSet, k)

print("最终质心：",centroids)
print('\n')
print("各个点到质心的距离：",clusterAssment)

showCluster(dataSet, k, centroids, clusterAssment)
