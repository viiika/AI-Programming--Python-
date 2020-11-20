


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

from numpy import *
from numpy import linalg as la
import cv2
import os

import time


# change the gray picture to row vector
def img2vector(filename):
    #print(filename)
    img = mpimg.imread(filename)
    #cv2.imshow('gray',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    # plt.imshow(img)
    # plt.show()
    # hog = Hog_descriptor(img, cell_size = 4, bin_size = 8)
    # vector, image = hog.extract()
    # return vector[0] # 1*32个特征值

    return img.reshape(1, -1)
    #return img


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
                #print(child_file)
                if count <rate *20:
                    trainSet[trainindex]=img2vector(os.path.join(child_path, child_file))
                    #print(trainindex)
                    #trainLabels.append(FileList)
                    trainLabels.append(kind-1)
                    trainindex+=1
                    count+=1
                else:
                    testSet[testindex]=img2vector(os.path.join(child_path, child_file))
                    #if kind-1 == 13:
                    #    print(testindex)
                    #testLabels.append(FileList)
                    #    print(child_file)
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
    #print(array2.shape)
    #time1=time.time()
    model3 = PCA(n_components=pcaK)  # 降维
    model3.fit(array2)  # 训练
    array3 = model3.fit_transform(array2)  # 降维后的数据  ###transform就会执行降维操作
    #time2=time.time()
    #print(time2-time1)
    trainSet, testSet = array3[:temp2], array3[temp2:]

    #model = cv2.face.FisherFaceRecognizer_create()
    #model = cv2.face.LBPHFaceRecognizer_create()
    model = cv2.face.EigenFaceRecognizer_create()
    print('created')
    #print(trainSet,trainLabels)
    #    model = cv2.face.EigenFaceRecognizer()
    # pip install opencv-contrib-python
    model.train(np.asarray(trainSet), np.asarray(trainLabels))
    print('trained')

    #model.predict(testSet,np.ndarray(testLabels))
    #while(1):
    #    a=1
    correct_count = 0
    test_number = testSet.shape[0]
    string = "test sample serial number: {0}, sample label: {1}, classify label: {2}------>correct?: {3}"
    for i in np.arange(test_number):
        (res,r)=model.predict(testSet[i])
        if res==testLabels[i]:
        #label = kNNClassify(testSet[i], trainSet, trainLabels, k=k)
        #if label == testLabels[i]:
            correct_count += 1
        print(string.format(i + 1, testLabels[i], res, res == testLabels[i]))
    #time3=time.time()
    #print(time3-time2)
    print("face recognization accuracy: {}%".format((correct_count / test_number) * 100))
    #return pcaK,(correct_count / test_number) * 100,time3-time2
    return (correct_count / test_number) * 100


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

#if __name__ == '__main__':
#      main(3,9)



def loadImageSet():
    data = readData('data/', 0.8)
    trainSet = data['trainSet']
    trainLabels = data['trainLabels']
    testSet = data['testSet']
    testLabels = data['testLabels']
    """
    FaceMat = mat(zeros((15, 98 * 116)))
    j = 0
    for i in os.listdir(add):
        if i.split('.')[1] == 'normal':
            try:
                img = cv2.imread(add + i, 0)
            except:
                print
                'load %s failed' % i
            FaceMat[j, :] = mat(img).flatten()
            j += 1
    """
    return trainSet,trainLabels,testSet,testLabels


def ReconginitionVector(selecthr=0.8):
    # step1: load the face image data ,get the matrix consists of all image
    trainSet, trainLabels, testSet, testLabels=loadImageSet()
    FaceMat = trainSet.T
    # step2: average the FaceMat
    avgImg = mean(FaceMat, 1)
    #avgImg = avgImg.reshape(200,180)
    #avgImg=avgImg.astype(np.int8)
    #print(avgImg)
    newavg=np.zeros((len(trainSet),200*180),dtype=int8)
    for i in range(len(trainSet)):
        newavg[i]=avgImg
    avgImg=newavg.T
    #cv2.imshow('avgImg',avgImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # step3: calculate the difference of avgimg and all image data(FaceMat)
    diffTrain = FaceMat - avgImg
    print(diffTrain.shape)
    print('OK')
    rres=mat(mat(diffTrain.T) * mat(diffTrain))
    # step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)
    eigvals, eigVects = linalg.eig(rres)
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:, eigSortIndex]  # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg[:,0], covVects, diffTrain,trainLabels,testSet,testLabels


def judgeFace(judgeImg, FaceVector, avgImg, diffTrain):
    #print('judge:',judgeImg.shape)
    #print('FaceVector:',FaceVector.shape)
    #print('avgImg',avgImg.shape)
    #print('diffTrain:',diffTrain.shape)
    diff = judgeImg.T - avgImg
    weiVec = FaceVector.T * diff# FaceVector  协方差矩阵的特征向量
    res = 0
    #resVal
    resVal = (array(weiVec - mat(np.asarray(FaceVector.T) * np.asarray(diffTrain[:, 0]))) ** 2).sum()+100
    print('judge')
    #print(FaceVector.T.shape)
    #print(diffTrain[:, 2].shape)
    for i in range(2432):#(36000,2432)
        TrainVec = mat(np.asarray(FaceVector.T) * np.asarray(diffTrain[:, i]))
        #print(TrainVec.shape)
        #print(weiVec.shape)
        #TrainVec = mat(np.asarray(FaceVector.T) * np.asarray((diffTrain.T)[i].T))
        if (array(weiVec - TrainVec) ** 2).sum() < resVal:
            res = i
            resVal = (array(weiVec - TrainVec) ** 2).sum()
    return res + 1


def myoutcome():
    avgImg, FaceVector, diffTrain ,trainLabels,testSet,testLabels= ReconginitionVector(selecthr=0.9)
    nameList = trainLabels

    count=0
    for i in range(len(testLabels)):
        if judgeFace(mat(testSet[i]).flatten(), FaceVector, avgImg, diffTrain) == int(testLabels[i]):
            count += 1
    print("face recognization accuracy: {}%".format((count / nameList) * 100))


if __name__ == '__main__':
    main(3,9)
    #myoutcome()

