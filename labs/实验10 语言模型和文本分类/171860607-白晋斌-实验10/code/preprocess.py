from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

file=open('stopwords.txt')
stopWords=file.read()
trainset=pd.read_csv('train_texts.txt',header=None)
trainlabel=pd.read_csv('train_labels.txt',header=None)
testset=pd.read_csv('test_texts.txt',header=None)
testlabel=pd.read_csv('test_labels.txt',header=None)
for i in range(7600):
    words = WordPunctTokenizer().tokenize(testset[0][i])
    #stopWords = set(stopwords.words('english'))
    wordsFiltered = []

    for w in words:
        if w not in stopWords and len(w)>2:
            wordsFiltered.append(w)

    testset[0][i]=wordsFiltered
    print(i)
testset.to_csv('test_texts2.csv', sep=',', header=False, index=False)

for i in range(120000):
    words = WordPunctTokenizer().tokenize(trainset[0][i])
    #stopWords = set(stopwords.words('english'))
    wordsFiltered = []

    for w in words:
        if w not in stopWords and len(w)>2:
            wordsFiltered.append(w)

    trainset[0][i]=wordsFiltered
    print(i)
trainset.to_csv('train_texts2.csv', sep=',', header=False, index=False)

print('write finished')