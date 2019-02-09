# author:gsy
# 李航 统计学习 例5.3 利用ID3算法建立决策树

from numpy import *
from math import log
import operator

# 计算经验熵：H(D)
def calcShannonEnt(dataset):
    numdataset = len(dataset)
    # print("numdataset",numdataset)
    labelCount = {}
    for data in dataset:
        curlabel = data[-1]  #获取最后一个类别
        if curlabel not in labelCount.keys():
            labelCount[curlabel] = 0
        labelCount[curlabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numdataset
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 构建数据集
# 四个特征 ；年龄，有工作，有自己的房子，信贷情况
# label:二分类变量 是/否
def creatDataSet():
    dataset = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否']]
    label = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset, label


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            redFeatVec = featVec[:axis]
            redFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(redFeatVec)
    return retDataSet

# 特征选择：遍历每个特征，选择信息增益最大的特征，作为特征切分
def choosebestFeaturnToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    # print("numFeature",numFeature)
    baseEntropy = calcShannonEnt(dataSet)  # H(D) 获取经验熵
    bestInfoGain = 0.0
    bestfeature = -1

    for i in range(numFeature):
        features = [example[i] for example in dataSet]
        uniqueVals = set(features)
        newEntropy = 0.0
        for value in uniqueVals:
            subdataSet = splitDataSet(dataSet, i, value)
            prob = len(subdataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subdataSet)
        InfoCain = baseEntropy - newEntropy
        if InfoCain > bestInfoGain:
            bestInfoGain = InfoCain
            bestfeature = i
    return bestfeature
    # return label[bestfeature]


def majority(classList):
    classcount = {}
    for vote in classList:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount += 1
    sortedclasscount = sorted(classcount, key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majority(classList)
    bestfeat = choosebestFeaturnToSplit(dataSet)
    bestfeatlabel = labels[bestfeat]
    myTree = {bestfeatlabel: {}}
    del (labels[bestfeat])
    featValues = [example[bestfeat] for example in dataSet]
    uniqualVals = set(featValues)
    for value in uniqualVals:
        myTree[bestfeatlabel][value] = createTree(splitDataSet(dataSet, bestfeat, value), labels)
    return myTree


dataSet, labels = creatDataSet()
# print(dataSet,labels)

shannonEnt = calcShannonEnt(dataSet)
print(shannonEnt)   # 求经验熵H(D)

bestfeature = choosebestFeaturnToSplit(dataSet)
print(bestfeature)

mytree = createTree(dataSet, labels)
print(mytree)
