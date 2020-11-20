from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.utils import plot_model
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Embedding
from keras.models import Sequential
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

import os

from collections import Counter
labeldic={'Business': 0, 'Sci/Tech': 1, 'Sports': 2, 'World': 3}
MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 200 # 词向量空间维度

def train():

    traintexts=pd.read_csv('train_texts.txt',header=None)
    trainlabels=pd.read_csv('train_labels.txt',header=None)
    testtexts=pd.read_csv('test_texts.txt',header=None)
    testlabels=pd.read_csv('test_labels.txt',header=None)
    traintext = traintexts[0].values.tolist()
    trainlabel = trainlabels[0].values.tolist()
    testtext = testtexts[0].values.tolist()
    testlabel = testlabels[0].values.tolist()
    text=traintext+testtext
    trainlabel = list(map(lambda x: labeldic[x], trainlabel))
    testlabel = list(map(lambda x: labeldic[x], testlabel))
    label=trainlabel+testlabel


    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    count_v0 = CountVectorizer()
    counts_all = count_v0.fit_transform(text)
    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_train = count_v1.fit_transform(traintext)
    print("the shape of train is " + repr(counts_train.shape))
    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_test = count_v2.fit_transform(testtext)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit(counts_train).transform(counts_train)
    test_data = tfidftransformer.fit(counts_test).transform(counts_test)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, trainlabel)
    preds = clf.predict(test_data)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(testlabel[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))



if __name__ == '__main__':
    train()