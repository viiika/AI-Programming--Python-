from sklearn.decomposition import PCA
import os
import cv2
import shutil
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

# change the gray picture to row vector
def img2vector(filename):
    #print(filename)
    img = mpimg.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    # plt.imshow(img)
    # plt.show()
    # hog = Hog_descriptor(img, cell_size = 4, bin_size = 8)
    # vector, image = hog.extract()
    # return vector[0] # 1*32个特征值

    return img.reshape(1, -1)


# 读取训练集和数据集
def readData(FilePath,rate):  #0-rate 训练集   rate-1 测试集
    trainingLen = int(3040 * rate)
    testingLen = int(3040 * (1 - rate))+1
    #print(trainingLen,testingLen)
    trainLabels = []
    trainSet = np.zeros((trainingLen, 200 * 180))  # 200*180
    testSet = np.zeros((testingLen, 200 * 180))
    testLabels = []
    trainindex=0
    testindex=0
    kind=0
    personName={}
    for FileList in os.listdir(FilePath):
        if FileList == '.DS_Store':
            os.remove(os.path.join(FilePath, FileList))
        else:
            personName[FileList]=kind
            #if kind ==13 or kind ==125 or kind ==146:
            #    print(kind,FileList)
            kind+=1
            child_path = os.path.join(FilePath, FileList)
            count=0
            for child_file in os.listdir(child_path):
                if count <rate *20:
                    trainSet[trainindex]=img2vector(os.path.join(child_path, child_file))
                    #print(trainindex)
                    #trainLabels.append(FileList)
                    trainLabels.append(kind-1)
                    trainindex+=1
                    count+=1
                else:
                    testSet[testindex]=img2vector(os.path.join(child_path, child_file))
                    if kind-1 == 13:
                        print(testindex)
                    #testLabels.append(FileList)
                        print(child_file)
                    testLabels.append(kind-1)
                    testindex+=1
                    count+=1
    print('finished readData!')
    return {'trainSet': trainSet, 'trainLabels': trainLabels,
            'testSet': testSet, 'testLabels': testLabels}






# 进行归一化处理 normalization
def maxmin_norm(array):
    """
    :param array: 每行为一个样本，每列为一个特征，且只包含数据，没有包含标签
    :return:
    """
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows, data_cols = data_shape
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


# KNN classifier
def kNNClassify(inX, dataSet, labels, k=3):
    """
    :param inX: 测试的样本的112*92灰度数据
    :param dataSet: 训练集
    :param labels: 训练集的标签列表('1' '2' ..... '40'共40类别)
    :param k: k值
    :return: 预测标签label
    distance是[5 50 149...]是测试样本距离每个训练样本的距离向量40 * 1
    """
    distance = np.sum(np.power((dataSet - inX), 2), axis=1)  # 计算欧几里得距离
    sortedArray = np.argsort(distance, kind="quicksort")[:k]

    # 给距离加入权重
    w = []
    for i in range(k):
        w.append((distance[sortedArray[k - 1]] - distance[sortedArray[i]]) \
                 / (distance[sortedArray[k - 1]] - distance[sortedArray[0]]))

    count = np.zeros(153)
    temp = 0
    for each in sortedArray:
        count[labels[each]] += 1 + w[temp]
        temp += 1

    label = np.argmax(count)  # 如果label中有多个一样的样本数，那么取第一个最大的样本类别
    return label




def main(k,pcaK):

    # read data
    data = readData('data/',0.8)
    trainSet = data['trainSet']
    trainLabels = data['trainLabels']
    testSet = data['testSet']
    testLabels = data['testLabels']

    # normalization
    temp = trainSet.shape[0]
    array = np.vstack((trainSet, testSet))
    normalized_array = maxmin_norm(array)
    trainSet, testSet = normalized_array[:temp], normalized_array[temp:]
    #降维
    temp2 = trainSet.shape[0]
    array2 = np.vstack((trainSet, testSet))
    print(array2.shape)
    time1=time.time()
    model3 = PCA(n_components=pcaK)  # 降维
    model3.fit(array2)  # 训练
    array3 = model3.fit_transform(array2)  # 降维后的数据  ###transform就会执行降维操作
    time2=time.time()
    print(time2-time1)
    trainSet, testSet = array3[:temp], array3[temp:]

    correct_count = 0
    test_number = testSet.shape[0]
    string = "test sample serial number: {0}, sample label: {1}, classify label: {2}------>correct?: {3}"
    for i in np.arange(test_number):
        label = kNNClassify(testSet[i], trainSet, trainLabels, k=k)
        if label == testLabels[i]:
            correct_count += 1
       # print(string.format(i + 1, testLabels[i], label, label == testLabels[i]))
    time3=time.time()
    print(time3-time2)
    print("face recognization accuracy: {}%".format((correct_count / test_number) * 100))
    return pcaK,(correct_count / test_number) * 100,time3-time1
    #return (correct_count / test_number) * 100


# verify the proper k
def selectK():
    x = list()
    y = list()
    for i in range(1, 11):
        x.append(i)
        y.append(main(i,10))
    plt.plot(x, y)
    plt.show()


def sunday2():
    pack=[]
    correct=[]
    thetime=[]
    for i in range(10):
        p,c,t=main(3,i)
        pack.append(p)
        correct.append(c)
        thetime.append(t)
    plt.figure()
    plt.plot(pack,correct,c='red',label='correct-rate')
    plt.ylabel('correct-rate')
    plt.xlabel('pca')
    plt.show()
    plt.figure()
    plt.plot(pack,thetime,c='blue',label='time')
    plt.ylabel('time')
    plt.xlabel('pca')
    plt.show()

if __name__ == '__main__':
    #selectK()
    sunday2()