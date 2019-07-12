import random
from multiprocessing import Queue, Process
import time
import multiprocessing
from multiprocessing import cpu_count
from numpy import *
import pandas as pd
import numpy as np
import numpy
from threading import Thread
# import random  
from random import sample 




def loadTrainDataSet():
    '''
    加载训练数据
    '''
    train_data1 = pd.read_csv('./data/train1.csv',header=None)
    train_data2 = pd.read_csv('./data/train2.csv',header=None)
    train_data3 = pd.read_csv('./data/train3.csv',header=None)
    train_data4 = pd.read_csv('./data/train4.csv',header=None)
    train_data5 = pd.read_csv('./data/train5.csv',header=None)
    
    label1 = pd.read_csv('./data/label1.csv',header=None)
    label2 = pd.read_csv('./data/label2.csv',header=None)
    label3 = pd.read_csv('./data/label3.csv',header=None)
    label4 = pd.read_csv('./data/label4.csv',header=None)
    label5 = pd.read_csv('./data/label5.csv',header=None)
    
    train = pd.concat([train_data1, train_data2, train_data3, train_data4, train_data5], sort=False)
    label = pd.concat([label1, label2, label3, label4, label5], sort=False)
    
    train = np.array(train)
    label = np.array(label)
    
    train = np.c_[train, label]
    return np.array(train)


def loadTestDataSet():
    '''
    加载测试数据
    '''
    test_data1 = pd.read_csv('./data/test1.csv',header=None)
    test_data2 = pd.read_csv('./data/test2.csv',header=None)
    test_data3 = pd.read_csv('./data/test3.csv',header=None)
    test_data4 = pd.read_csv('./data/test4.csv',header=None)
    test_data5 = pd.read_csv('./data/test5.csv',header=None)
    test_data6 = pd.read_csv('./data/test6.csv',header=None)
    test_data = pd.concat([test_data1, test_data2, test_data3, test_data4, test_data5, test_data6], sort=False)
    test_data = np.array(test_data)
    return test_data



class RegressionTree:
    def __init__(self, min_increase_gain = 1, min_samples_split = 100):
        '''
        min_increase_gain: 一个节点分裂以后误差的最小减少值，当值小于这个值的时候停止分裂
        min_samples_split: 每个叶子的最小样本数
        '''
        self.min_increase_gain = min_increase_gain
        self.min_samples_split = min_samples_split
    
    def binSplitDataSet(self, dataSet, feature, value):
        ''' 
        划分数据集,
        这里进行了cache的优化，使用方案1的时候跑得太慢了，每次都要遍历全部的特征
        改变以后，由于传进来的dataSet是按照feature排序的，所以当一个大于当前的value的时候，我们就可以将数据集划分开来了
        '''
        '''
        方案1
        dataSet00 = []
        dataSet11 = []
        for i in range(len(dataSet)):
            if dataSet[i][feature] > value:
                dataSet00.append(dataSet[i])
            else:
                dataSet11.append(dataSet[i])
             
        return np.array(dataSet00), np.array(dataSet11)
        '''
        # 方案2
        index = -1
        dataSet0 = []
        dataSet1 = []
        for i in range(len(dataSet)):
            if dataSet[i][feature] > value:
                index = i
                break
        
        if index == 0:
            return dataSet, np.array(dataSet0)
        elif index == -1:
            return np.array(dataSet0), dataSet
        else:
            dataSet1 = dataSet[0:index]
            dataSet0 = dataSet[index:]
            return np.array(dataSet0),np.array(dataSet1)
    
    def leafValue(self, dataSet):
        '''
        求叶子的均值，作为最后预测的结果
        '''
        return mean(dataSet[:,-1])
    
    
    def leafVar(self, dataSet):
        '''
        求方差，用于选取最佳分割点
        '''
        return var(dataSet[:,-1]) * shape(dataSet)[0]

        
    def chooseBestSplit(self, dataSet):
        '''
        选取最佳分割点，在对数据进行分割之前，先对特征进行排序，
        以减少cache与内存的换页，可以有效减少训练时间
        
        '''
        # 如果叶子的所有值相等，不需要继续划分
        if len(set(dataSet[:,-1].T.tolist())) == 1:
            return None, self.leafValue(dataSet)
        m, n = shape(dataSet)
        # 计算父节点方差
        parentVar = self.leafVar(dataSet)
        bestVar = inf
        bestIndex = 0
        bestValue = 0
        # 随机选取分割特征
        # 设置随机种子，在多进程的情况下，需要设置，否则生成的随机数不是严格意义上的随机数
        # 随机选择特征的1/3来进行分割
        random_features = sample(range(0, n - 1),int((n-1)/3))
        
        for featIndex in random_features:
            # 对特征进行排序，减少换页
            temp = [x[featIndex] for x in dataSet]
            index = np.argsort(temp ,axis=0)
            dataSet = dataSet[index]
            temp.sort()
            temp = list(set(temp))
            t = []
            for i in range(len(temp)):
                if i == 0:
                    tag = temp[i]
                    t.append(temp[i])
                else:
                    if temp[i] - tag > 1:
                        t.append(temp[i])
                        tag = temp[i]
            temp = t
            for splitVal in temp:
                dataSet0, dataSet1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                # 当分割后的叶子小于叶子最少样本数时，不需要分割
                if (shape(dataSet0)[0] < self.min_samples_split) or (shape(dataSet1)[0] < self.min_samples_split): 
                    continue
                # 分割后的方差
                newVar = len(dataSet0)/len(dataSet)*self.leafVar(dataSet0) + len(dataSet1)/len(dataSet)*self.leafVar(dataSet1)
                # 分割后的方差与分割前的进行比较，去较好值
                if newVar < bestVar: 
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestVar = newVar
        # 如果增益过小，不需要分割
        if (parentVar - bestVar) < self.min_increase_gain: 
            return None, self.leafValue(dataSet)
        
        temp = [x[bestIndex] for x in dataSet]
        # temp = dataSet[:,featIndex]
        index = np.argsort(temp ,axis=0)
        dataSet = dataSet[index]
        dataSet0, dataSet1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (shape(dataSet0)[0] < self.min_samples_split) or (shape(dataSet1)[0] < self.min_samples_split):
            return None, self.leafValue(dataSet)
        
        return bestIndex, bestValue
    
    def createTree(self, dataSet, deep = 20):
        '''
        综合上面过程，建一棵完整的树
        递归实现
        '''
        deep -= 1
        if deep == 0:
            return self.leafValue(dataSet)
        feat, val = self.chooseBestSplit(dataSet)
        if feat == None: 
            return val
        retTree = {}
        retTree['featureIndex'] = feat
        retTree['spiltValue'] = val
        temp = [x[feat] for x in dataSet]
        index = np.argsort(temp ,axis=0)
        dataSet = dataSet[index]
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree['left'] = self.createTree(lSet, deep)
        retTree['right'] = self.createTree(rSet, deep)
        return retTree


class RandomForestRegressor:
    def __init__(self, n_estimators = 2, max_depth = 20, sample_ratio = 0.1, min_sample_spilt = 100, process_num = 8, min_increase_gain = 1):
        '''
        n_estimators：树的个数
        max_depth：树的最大深度 
        sample_ratio：取样时候的比例
        min_sample_spilt：叶子最小样本数
        process_num：使用的进程数量
        '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.forests = []
        self.sample_ratio = sample_ratio
        self.min_sample_spilt = min_sample_spilt
        self.process_num = process_num
        self.min_increase_gain = min_increase_gain
        
    def predict_one_tree(self, tree, data):
        return predict(data)
    
    def fit(self, dataSet):
        # 多进程操作
        pool = multiprocessing.Pool(processes = self.process_num)
        start = time.time()
        result = []
        for i in range(self.n_estimators):
            result.append(pool.apply_async(self.build_tree_mulprocess, args=(i, dataSet,)))
        
        pool.close()
        pool.join()
        end = time.time()
        print('time:', (end - start))
        for r in result:
            self.forests.append(r.get())

    def build_tree_mulprocess(self, i, dataSet):
        """
        用于建树的进程
        """
        print('Building tree ', i, ' ...')
        sample_dataSet = self.get_sample_dataSet(dataSet, i)
        t = RegressionTree(min_increase_gain=self.min_increase_gain, min_samples_split=self.min_sample_spilt)
        tree = t.createTree(sample_dataSet, self.max_depth)
        print('Build tree ', i, ' end')
        return tree

    def get_sample_dataSet(self, dataSet, i):
        '''
        有放回选取数据集，
        smple_ratio是选取的数据集占原样本的比例
        '''
        print('Getting data for tree ', i)
        sample_size = round(len(dataSet) * self.sample_ratio)
        sample_dataSet = []
        random_indexs = []
        while len(random_indexs) < sample_size:
            random_indexs.append(random.randint(0, len(dataSet)-1))
        
        # 排序再取，避免过多分页
        random_indexs.sort()
        for j in random_indexs:
            sample_dataSet.append(np.array(dataSet[j]).reshape(14))
        print('Get data for tree ', i, ' end')
        return np.array(sample_dataSet)
    
    def getForest(self):
        return self.forests
    
    def predict(self, data):
        '''
        预测森林的效果
        '''
        result = []
        for j in range(len(data)):
            if j%10000 == 0:
                print('[{}] of [{}]'.format(j, len(data)))
            s = 0
            for i in range(len(self.forests)):
                temp = self.forests[i]
                while 1:
                    if type(temp) == np.float64:
                        s += temp
                        break
                    index = temp['featureIndex']
                    d = data[j][index]
                    if d > temp['spiltValue']:
                        temp = temp['left']
                    else:
                        temp = temp['right']
            result.append(s/len(self.forests))
        return np.array(result)


train_data = loadTrainDataSet()
test_data = loadTestDataSet()

r = RandomForestRegressor(n_estimators = 100, max_depth = 10, sample_ratio=0.01, min_sample_spilt = 1000, process_num = 8)
r.fit(train_data)

r.getForest()

result = r.predict(test_data)

df = pd.DataFrame({'Predicted':result})
df.index += 1
df.to_csv('./data/my_sub.csv', index = True, index_label = 'id')