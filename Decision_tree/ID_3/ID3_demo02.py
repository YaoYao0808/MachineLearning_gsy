# ID3构建决策树 非最终版


# print(type(dataset))  <class 'list'>
# print(dataset[0])  dataset[0] dataset[1] dataset[2] 是获取每一个样本数据

# list 操作
# (1).负数索引
"""
print("[0]:",dataset[0])
print("[1:4]:",dataset[1:4])
print("[-3]:",dataset[-3])
"""
# (2).增加元素
"""
dataset.append()
dataset.insert()
dataset.extend()
"""
# (3).list搜索
"""
print(dataset.index(['青年', '否', '否', '一般', '否']))   索引找到为0
print(dataset.index(['老年', '是', '否', '非常好', '是'])) 索引找到为13
print(['青年', '否', '否', '一般', '否'] in dataset)   返回true
"""
# (4).list删除元素
"""
dataset.remove()   会删除首次出现的第一个值,没有找到值则会引发异常
dataset.pop()      删除 list 的最后一个元素, 然后返回删除元素的值
"""
# (5).join 连接和分割字符串
# 参考链接：http://www.runoob.com/python3/python3-list-operator.html

# 2.求经验熵H(D)
"""
from math import log

numdataset = len(dataset)
print("numdataset",numdataset)   15个数据集
labelCount = {}
print(type(labelCount))  <class 'dict'> 字典类型

for data in dataset:  #循环遍历结果
    curlabel = data[-1]  #list负数索引，获取最后一个二分类变量 是/否
    if curlabel not in labelCount.keys():
        labelCount[curlabel] = 0
    labelCount[curlabel] += 1
shannonEnt = 0.0

print("labelCount",labelCount)  labelCount {'否': 6, '是': 9}

# 此处求经验熵H(D):-9/15*log(9/15)-6/15*log(6/15)=9709505944546686
for key in labelCount:
    prob = float(labelCount[key]) / numdataset
    shannonEnt -= prob * log(prob, 2)

print(shannonEnt)  0.9709505944546686=0.971

"""
# 1。准备数据集   二分类变量  四个特征
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

# 求得经验熵整合为一个方法 H(D)
from math import log

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


# 划分出某特征=value值的样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            redFeatVec = featVec[:axis]
            redFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(redFeatVec)
    return retDataSet

"""
#  获取最佳特征
# print("len(dataset[0]):",len(dataset[0]))  5  获取列数

numFeature = len(dataset[0]) - 1   #列数-1为特征数
print("numFeature",numFeature)
baseEntropy = calcShannonEnt(dataset)  # H(D) 获取经验熵
bestInfoGain = 0.0
bestfeature = -1

for i in range(numFeature):  #依次遍历各个特征获取信息增益最小的特征
    # 依次获取每个样本的第一个特征、第二个特征、第三个特征、第四个特征，最后返回四个列表变量
    features = [example[i] for example in dataset]   #['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年']
    # print(features)
    # print(type(features))  <class 'list'>
    uniqueVals = set(features)
    # print(uniqueVals)  {'青年', '老年', '中年'}
    # print(type(uniqueVals))  <class 'set'>
    # set()去除重复，且进行类型转换  list->set
    newEntropy = 0.0
    for value in uniqueVals:
        # print(value)
        subdataSet = splitDataSet(dataset, i, value)
        # print(subdataSet)
        prob = len(subdataSet) / float(len(dataset))
        newEntropy += prob * calcShannonEnt(subdataSet)
    InfoCain = baseEntropy - newEntropy
    if InfoCain > bestInfoGain:
        bestInfoGain = InfoCain
        bestfeature = i

print("bestfeature",bestfeature)  #2 表示最佳划分特征为第三个
"""
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

import operator

def majority(classList):
    classcount = {}
    for vote in classList:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount += 1
    sortedclasscount = sorted(classcount, key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]

# 3.构建决策树  递归
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 获取最后一列二分类变量
    print(type(classList))
    # print(classList.count(classList[0]))
    # 样本都属于一类的情况
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 只剩下一个特征
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


mytree = createTree(dataset, label)
print(mytree)   # {'有自己的房子': {'否': {'有工作': {'否': '否', '是': '是'}}, '是': '是'}}



