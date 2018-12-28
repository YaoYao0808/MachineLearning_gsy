#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    ''' 加载数据
    '''
    X, Y = [], []
    with open(filename, 'r') as f:
        for line in f:
            splited_line = [float(i) for i in line.split()]
            x, y = splited_line[: -1], splited_line[-1]
            X.append(x)
            Y.append(y)
    X, Y = np.matrix(X), np.matrix(Y).T
    return X, Y

def standarize(X):
    ''' 中心化 & 标准化数据 (零均值, 单位标准差)
    '''
    std_deviation = np.std(X, 0)  #计算每一列的标准差
    mean = np.mean(X, 0)          #计算每一列的均值
    return (X - mean)/std_deviation

# 求解w回归系数
def std_linreg(X, Y):
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:  #np.linalg.det() 求方阵的行列式值,此方法要求是方阵
        print('xTx is a singular matrix')
        return
    # .I：求逆  返回theta参数  正规方程组解
    return xTx.I*X.T*Y

def get_corrcoef(X, Y):
    # X Y 的协方差
    cov = np.mean(X*Y) - np.mean(X)*np.mean(Y)
    return cov/(np.var(X)*np.var(Y))**0.5

if '__main__' == __name__:
    # 加载数据
    X, Y = load_data('abalone.txt')
    X, Y = standarize(X), standarize(Y)
    w = std_linreg(X, Y)
    Y_prime = X*w

    print('w: {}'.format(w))

    # 计算相关系数
    corrcoef = get_corrcoef(np.array(Y.reshape(1, -1)),
                            np.array(Y_prime.reshape(1, -1)))
    print('Correlation coeffient: {}'.format(corrcoef))

    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    ## 绘制数据点
    #x = X[:, 1].reshape(1, -1).tolist()[0]
    #y = Y.reshape(1, -1).tolist()[0]
    #ax.scatter(x, y)

    ## 绘制拟合直线
    #x1, x2 = min(x), max(x)
    #y1 = (np.matrix([1, x1])*w).tolist()[0][0]
    #y2 = (np.matrix([1, x2])*w).tolist()[0][0]
    #ax.plot([x1, x2], [y1, y2], c='r')

    #plt.show()

