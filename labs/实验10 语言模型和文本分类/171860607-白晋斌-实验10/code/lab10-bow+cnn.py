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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)#首先用Tokenizer的 fit_on_texts 方法学习出文本的字典
    #print(tokenizer.word_index) #word_index 就是对应的单词和数字的映射关系dict
    sequences = tokenizer.texts_to_sequences(text)#通过这个dict可以将每个string的每个词转成数字，可以用texts_to_sequences
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' %len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)#通过padding的方法补成同样长度，在用keras中自带的embedding层进行一个向量化
    labels = to_categorical(np.array(label))
    #print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:100000]
    y_train = labels[:100000]
    x_val = data[100000:120000]
    y_val = labels[100000:120000]
    x_test = data[120000:]
    y_test = labels[120000:]
    print('train docs: ' + str(len(x_train)))
    print('val docs: ' + str(len(x_val)))
    print('test docs: ' + str(len(x_test)))

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.2))
    model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save('word_vector_cnn.h5')
    print(model.evaluate(x_test, y_test))

def test():
    traintexts = pd.read_csv('train_texts.txt', header=None)
    trainlabels = pd.read_csv('train_labels.txt', header=None)
    testtexts = pd.read_csv('test_texts.txt', header=None)
    testlabels = pd.read_csv('test_labels.txt', header=None)
    traintext = traintexts[0].values.tolist()
    trainlabel = trainlabels[0].values.tolist()
    testtext = testtexts[0].values.tolist()
    testlabel = testlabels[0].values.tolist()
    text = traintext + testtext
    trainlabel = list(map(lambda x: labeldic[x], trainlabel))
    testlabel = list(map(lambda x: labeldic[x], testlabel))
    label = trainlabel + testlabel

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)  # 首先用Tokenizer的 fit_on_texts 方法学习出文本的字典
    # print(tokenizer.word_index) #word_index 就是对应的单词和数字的映射关系dict
    sequences = tokenizer.texts_to_sequences(text)  # 通过这个dict可以将每个string的每个词转成数字，可以用texts_to_sequences
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 通过padding的方法补成同样长度，在用keras中自带的embedding层进行一个向量化
    labels = to_categorical(np.array(label))
    # print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:100000]
    y_train = labels[:100000]
    x_val = data[100000:120000]
    y_val = labels[100000:120000]
    x_test = data[120000:]
    y_test = labels[120000:]
    print('train docs: ' + str(len(x_train)))
    print('val docs: ' + str(len(x_val)))
    print('test docs: ' + str(len(x_test)))

    from keras.models import load_model

    print("Using loaded model to predict...")
    model = load_model("word_vector_cnn.h5")
    print(model.evaluate(x_test, y_test))

if __name__ == '__main__':
    train()
    #test()