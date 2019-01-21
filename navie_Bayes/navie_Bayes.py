# 代码数据基于李航机器学习中P50页的例4.1

# coding: utf-8 or # -*- coding: utf-8 -*-
# author=gsy
import numpy as np

# 获得特征向量可能值  ['l', 1, 2, 3, 'm', 's']
def createWordList(data):
    wordSet = set([])
    for document in data:
        wordSet = wordSet | set(document)
    return list(wordSet)


# 将多维数据转化为一维向量，方便计算
def word2Vec(wordList, inputWord):
    returnVec = [0] * len(wordList)
    for word in inputWord:
        if word in wordList:
            returnVec[wordList.index(word)] = 1
    return returnVec


# 训练函数，根据给定数据和标签，计算概率
def train(trainMatrix, trainLabels):
    # 此处要学会获取矩阵行列的方法  len(trainMatrix)为行数，trainMatrix[0]为列数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # print("numTrainDocs:",numTrainDocs)  15  此处为样本数
    # print("numWords",numWords)  6   此外为特征向量取值总数

    pAbusive = (sum(trainLabels) + 1.0) / (float(numTrainDocs) + 2.0 * 1.0)  # 10/17
    p0Num = np.ones(numWords)    #条件概率下，对于正负样本均有6个学习参数，numWords个
    p1Num = np.ones(numWords)

    # 此处是贝叶斯估计,取值为2得拉普拉斯平滑得情况
    # 拉普拉斯下 01样本条件概率的分母    9和12
    p0Denom = 3.0 + len(trainLabels) - sum(trainLabels)  #3*1+6(0样本数)    每个特征有3种取值   注意：若两个特征的取值数目不一样，则要分开求
    p1Denom = 3.0 + sum(trainLabels)

    for i in range(numTrainDocs):
        if trainLabels[i] == 1:  # 正样本
            p1Num += trainMatrix[i]
        else:                   #负样本
            p0Num += trainMatrix[i]

    # p0Num: [2. 4. 3. 2. 4. 3.]   求得先验概率--条件概率，0样本下的6个分子
    # p1Num: [3. 2. 4. 5. 5. 5.]   求得先验概率--条件概率，1样本下的6个分子

    p0Vect = np.log(p0Num / p0Denom)   #此处采用Log函数进行转换，因此所有的条件概率都是0-1的小数
    p1Vect = np.log(p1Num / p1Denom)

    return p0Vect, p1Vect, pAbusive


# 分类函数
def classify(vec2Clssify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Clssify * p1Vect) + np.log(pClass1)
    p0 = sum(vec2Clssify * p0Vect) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 特征说明：两个属性特性，第一个[1,2,3],第二个['s','m','l']，预测结果y为二分类变量[0,1]
def main():
    data = [[1, 's'], [1, 'm'], [1, 'm'], [1, 's'], [1, 's'], [2, 's'], [2, 'm'], [2, 'm'], [2, 'l'], [2, 'l'],
            [3, 'l'], [3, 'm'], [3, 'm'], [3, 'l'], [3, 'l']]
    labels = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    wordList = createWordList(data)
    # print("获得特征向量的可能值",wordList)  ['l', 1, 2, 3, 'm', 's']

    dataMatrix = []
    for item in data:
        dataMatrix.append(word2Vec(wordList, item))
    # [[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0]]
    # print("将多维数据转换为一维向量后：",dataMatrix)

    p0, p1, pAB = train(dataMatrix, labels)
    # p0[-0.81093022 - 1.09861229 - 0.81093022 - 1.5040774 - 1.5040774 - 1.09861229]
    # p1[-1.38629436 - 1.09861229 - 1.79175947 - 0.87546874 - 0.87546874 - 0.87546874]
    # pAB 0.5882352941176471

    goal = [3, 'l']
    # 先将预测的特征向量转换为一维向量
    wordVec = np.array(word2Vec(wordList, goal))
    print(classify(wordVec, p0, p1, pAB))


if __name__ == '__main__':
    main()
