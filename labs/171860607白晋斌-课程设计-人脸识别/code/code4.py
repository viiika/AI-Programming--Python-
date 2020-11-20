import os
import cv2
import numpy as np
import json
import pickle
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128
    file_count = 0
    labbel={}
    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        if child_dir == '.DS_Store':
            continue
        child_path = os.path.join(path, child_dir)

        for child_dir2 in os.listdir(child_path):
             if child_dir2 == '.DS_Store':
                 continue
             child_path2 = os.path.join(child_path,child_dir2)
             num=20
             for dir_image in os.listdir(child_path2):
                 if dir_image=='.DS_Store':
                     continue
                 if dir_image[-3:]=='gif':
                     continue
                 num-=1
                 #if num>0:
                 #    print(child_path2)
                 file_count += 1
                 #print(dir_image)
                 if endwith(dir_image,'jpg'):
                    img = cv2.imread(os.path.join(child_path2, dir_image))
                    #print(img)
                    resized_img = cv2.resize(img, (200, 180))
                    recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                    img_list.append(recolored_img)
                    label_list.append(dir_counter)
                    labbel[dir_counter]=dir_image[:-6]
             dir_counter += 1
    with open('label.txt', 'wb') as file:
        #file.write(json.dumps(labbel))
        pickle.dump(labbel, file)
        # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)
    return img_list, label_list, dir_counter
    #return img_list,label_list,dir_counter,file_count


#useless
#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

#建立一个用于存储和格式化读取训练数据的类
class DataSet(object):
   def __init__(self,path):
       self.num_classes = None
       self.X_train = None
       self.X_test = None
       self.Y_train = None
       self.Y_test = None
       self.img_size = 128
       self.img_X=200
       self.img_Y=180
       self.extract_data(path) #在这个类初始化的过程中读取path下的训练数据

   def extract_data(self,path):
        #根据指定路径读取出图片、标签和类别数
        imgs,labels,counter = read_file(path)

        #将数据集打乱随机分组
        X_train,X_test,y_train,y_test = train_test_split(imgs,labels,test_size=0.2,random_state=random.randint(0, 100))

        #重新格式化和标准化
        # 本案例是基于thano的，如果基于tensorflow的backend需要进行修改
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_X, self.img_Y)/255.0
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_X, self.img_Y) / 255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        #将labels转成 binary class matrices
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)

        #将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

   def check(self):
       print('num of dim:', self.X_test.ndim)
       print('shape:', self.X_test.shape)
       print('size:', self.X_test.size)

       print('num of dim:', self.X_train.ndim)
       print('shape:', self.X_train.shape)
       print('size:', self.X_train.size)
#建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "model.h5"   #模型进行存储和读取的地方
    #IMAGE_SIZE = 128    #模型接受的人脸图片一定得是128*128的  #in fact is 180*200

    def __init__(self):
        self.model = None

    #读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self,dataset):
        self.dataset = dataset

    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()
        plot_model(self.model,to_file='cnn_model.png',show_shapes=True)

    #进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adam',  #有很多可选的optimizer，例如RMSprop,Adagrad，你也可以试试哪个好，我个人感觉差异不大
            loss='categorical_crossentropy',  #你可以选用squared_hinge作为loss看看哪个好
            metrics=['accuracy'])

        #epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train,self.dataset.Y_train,epochs=20,batch_size=20)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    #需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self,img):
        img = img.reshape((1, 1, 200, 180))
        #img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)  #测算一下该img属于某个label的概率
        max_index = np.argmax(result) #找出概率最高的

        return max_index,result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率


if __name__ == '__main__':
    dataset = DataSet('../faces94')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    #model.summary()
    #plot_model(model,to_file='cnn_model.png')
    model.train_model()
    model.evaluate_model()
    model.save()














