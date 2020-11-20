'''
CLASS1：
数据表示和降维
    PCA 主成分分析法
    kPCA
    流形学习
Reference：
https://blog.csdn.net/Reticent_Man/article/details/82633214
https://blog.csdn.net/u013082989/article/details/53792010
参数解释：

n_components:  我们可以利用此参数设置想要的特征维度数目，可以是int型的数字，也可以是阈值百分比，如95%，让PCA类根据样本特征方差来降到合适的维数，也可以指定为string类型，MLE。
copy： bool类型，TRUE或者FALSE，是否将原始数据复制一份，这样运行后原始数据值不会改变，默认为TRUE。
whiten：bool类型，是否进行白化（就是对降维后的数据进行归一化，使方差为1），默认为FALSE。如果需要后续处理可以改为TRUE。
explained_variance_： 代表降为后各主成分的方差值，方差值越大，表明越重要。
explained_variance_ratio_： 代表各主成分的贡献率。
inverse_transform()： 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)。
'''
'''
#homework1
#coding=utf-8
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
model = PCA(n_components=2)   #降到2维
model.fit(X)                  #训练
newX=model.fit_transform(X)   #降维后的数据  ###transform就会执行降维操作
# PCA(copy=True, n_components=2, whiten=False)
print(model.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据
###数据恢复并作图
Ureduce = model.components_     # 得到降维用的U矩阵
x_rec = np.dot(newX,Ureduce)       # 数据恢复
'''
'''
CLASS2：
有监督学习
    回归与分类
评估方法
回归模型
    线性回归
分类模型
    罗技斯蒂(logistic)回归

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('data_akbilgic.csv')
#print(df.shape)
"""
print(df.describe())
print(df.corr())#相关系数 默认为pearson相关系数
print(df.cov())#协方差
print(df.corr(method='spearman'))
print(df.corr(method='kendall'))
#绘制相关性矩阵图
plt.figure()
plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True, vmax=1, square=True, cmap="Reds")
plt.show()
#绘制折线图
df.plot()
plt.show()
"""
#df.set_index(["date"], inplace=True)
#线性回归
#data=df.drop(columns='date')
data=df.values.transpose()
#print(data)
#print(df)
#print(data)
#将数据分为训练diabetes_X_train /测试集diabetes_X_test
#500train
#随机抽取样本
col_rand_array = np.arange(data.shape[1])
np.random.shuffle(col_rand_array)
col_rand = data[:,col_rand_array[0:500]]
train=col_rand
col_rand = data[:,col_rand_array[0:100]]
test=col_rand
X_train=np.delete(train, [0,1], axis=0).transpose()

Y_train=train[[1]].transpose()
X_test=np.delete(test, [0,1], axis=0).transpose()
Y_test=test[[1]].transpose()

regr = LinearRegression().fit(X_train, Y_train)
pred=regr.predict(X_test)

# 查看系数
print('Coefficients: \n', regr.coef_)
# 查看均方误差
print("Mean squared error: %.2f"%mean_squared_error(Y_test, pred))
# 解释方差分数:1是完美的预测
print('Variance score: %.2f' %r2_score(Y_test, pred))

#图输出
plt.scatter(Y_test, pred, color='blue', linewidth=3)
X=np.linspace(-0.05,0.07,10)
plt.plot(X,X,color='green')
#plt.xticks(())
#plt.yticks(())
plt.show()
