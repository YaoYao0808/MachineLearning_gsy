import numpy as np

class SingleDecisionTree:   #决策树桩
    def __init__(self, axis=0, threshold = 0, flag = True):
        self.axis = axis
        self.threshold = threshold
        self.flag = flag #flag=True, x>=threshold=1, 否则为-1   判断是否为边界1?
    #     比如是x>=2.5为1，还是x>=2.5为-1，所以需要这个flag标志来说明是哪种情况

    def preditct(self, x):
        print("into Single predict")
        if (self.flag == True):
            return -1 if x[self.axis] >= self.threshold else 1
        else:
            return 1 if x[self.axis] >= self.threshold else -1

    def preditctArr(self, dataSet):
        # print("into Single preditctArr")
        result = list()
        for x in dataSet:
            if (self.flag == True):
                result.append(-1 if x[self.axis] >= self.threshold else 1)
            else:
                result.append(1 if x[self.axis] >= self.threshold else -1)
        return result

class Adaboost:
    def train(self, dataSet, labels):
        N = np.array(dataSet).shape[0]  # 样本总数   10
        M = np.array(dataSet).shape[1]  # 样本维度   1
        # print("样本总数:",N)
        # print("样本维度:", M)
        self.funList = list()  # 存储alpha和决策树桩
        D = np.ones((N, 1)) / float(N)  # (1)数据权值分布  初始化D中的10个权值均为0.1
        # 得到基本分类器 开始
        L = 0.5
        minError = np.inf  # 初始化误差大小为最大值（因为要找最小值）
        minTree = None  # 误差最小的分类器
        while minError > 0.01:
            for axis in range(M): #遍历每个样本维度
                min = np.min(np.array(dataSet)[:, axis])  # 需要确定阈值的最小值
                max = np.max(np.array(dataSet)[:, axis])  # 需要确定阈值的最大值
                for threshold in np.arange(min, max, L):  # 左开右闭  切分点[1,1.5,2,2.5,3,3.5,4,4.5......]
                    tree = SingleDecisionTree(axis=axis, threshold=threshold, flag=True)  # 决策树桩
                    # 此处是在for循环内部,选取分类误差率最低时的阈值,如p140的v=2.5
                    em = self.calcEm(D, tree, dataSet, labels)  # 误差率  def calcEm(self, D, Gm, dataSet, labels):
                    if (minError > em):  # 选出最小的误差，以及对应的分类器
                        minError = em
                        minTree = tree
                    tree = SingleDecisionTree(axis=axis, threshold=threshold, flag=False)  # 同上，不过flag的作用要知道  flag=True, x>=threshold=1, 否则为-1   其实此处flag=False也可以
                    em = self.calcEm(D, tree, dataSet, labels)
                    if (minError > em):
                        minError = em
                        minTree = tree
            alpha = (0.5) * np.log((1 - minError) / float(minError))  # p139(8.2)
            self.funList.append((alpha, minTree))  # 把alpha和分类器写到列表
            D = np.multiply(D, np.exp(np.multiply(-alpha * np.array(labels).reshape((-1, 1)),
                                                  np.array(minTree.preditctArr(dataSet)).reshape((-1, 1))))) / np.sum(
                np.multiply(D, np.exp(np.multiply(-alpha * np.array(labels).reshape((-1, 1)),
                                                  np.array(minTree.preditctArr(dataSet)).reshape(
                                                      (-1, 1))))))  # 对应p139的公式(8.4)
    def predict(self, x):   #预测方法
        sum = 0
        for fun in self.funList:    #书上最终分类器的代码
            alpha = fun[0]
            tree = fun[1]
            sum += alpha * tree.preditct(x)
        return 1 if sum > 0 else -1

    def calcEm(self, D, Gm, dataSet, labels):    #计算误差  D={w11,w12,w13...}，D中存储的是权值分布
        # value = list()
        # value值用来存储G(m)分类器的结果和实际结果是否相等，若相等，则值为0，若不相等，则值为1。
        value = [0 if Gm.preditct(row) == labels[i] else 1 for (i, row) in enumerate(dataSet)]
        # reshape((-1, 1):不清除数据的行数,只想让数据变成一列
        error=np.sum(np.multiply(D, np.array(value).reshape((-1, 1))))
        print("分类误差率:",error)  #p138页em的求法
        return np.sum(np.multiply(D, np.array(value).reshape((-1, 1))))

if __name__ == '__main__':
    dataSet = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]  #例8.1的数据集
    labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    # dataSet = [[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2], [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]]    #p153的例子
    # labels = [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1]
    # dataSet = [[1, 2], [1, 3], [2, 4], [3, 3], [3, 4], [4, 0.5], [5, 1.5], [5, 3], [5, 4]]    #练习题第一题的例子
    # labels = [-1, -1, -1, 1, 1, -1, 1, 1, 1]
    adaboost = Adaboost()
    adaboost.train(dataSet, labels)
    for x in dataSet:
        print(adaboost.predict(x))
    print("预测结果：",adaboost.predict([1, 3, 2]))


"""
1.np.arange()的使用
np.arange(1, 5, .5)    在1-5区间内，左闭右开，以0.5为步长
array([ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])

2.np.multiply()
各个维度对应相乘  
"""