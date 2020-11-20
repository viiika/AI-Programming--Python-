import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('wdbc.csv',header=None)
#print(df.)
#print(df)
data=df.iloc[:,2:]
""""
kmeans = KMeans(n_clusters=2, random_state=10).fit(data)
data['jllable']=kmeans.labels_
df_count_type=data.groupby('jllable').apply(np.size)
"""

#data=df.drop([0,1],axis=1)
model = PCA(n_components=2)   #降到2维
model.fit(data)                  #训练
newX=model.fit_transform(data)   #降维后的数据  ###transform就会执行降维操作
print(model.explained_variance_ratio_)  #输出贡献率
#print(newX)                  #输出降维后的数据
print(newX)



#ddd=pd.DataFrame(newX)
pltfig=df.iloc[:,1:2]
pd2=pd.DataFrame(newX)
pltfig=pd.concat([pltfig,pd2],ignore_index=True,axis=1)
print(pltfig)
colors = {'M':'red', 'B':'blue'}
fig, ax = plt.subplots()
grouped = pltfig.groupby(0)
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x=1, y=2, label=key, color=colors[key])
    #print(key)
plt.show()


while(1):
    a=1
#plt.figure()
#plt.scatter(newX[0],newX[1])
#plt.show()

#降维成两维然后聚类
"""
d = newX[data['jllable'] == 0]
plt.plot(d[0], d[1], 'r.')
d = newX[data['jllable'] == 1]
plt.plot(d[0], d[1], 'go')

plt.show()
"""
origdata=df.values.transpose()
#col_rand_array = np.arange(origdata.shape[1])
#np.random.shuffle(col_rand_array)
#col_rand = origdata[:,col_rand_array[0:500]]
#train=col_rand
train=origdata[:,:-50]
#col_rand = origdata[:,col_rand_array[0:100]]
#test=col_rand
test=origdata[:,-50:]
X_train=np.delete(train, [0,1], axis=0).transpose()
Y_train=train[[1]].transpose()
X_test=np.delete(test, [0,1], axis=0).transpose()
Y_test=test[[1]].transpose()
a=Y_train.shape
Y_train=np.apply_along_axis(lambda x:0 if x=='M' else 1, 1,Y_train)
#print(Y_train.type)
#Y_train.apply()
#Y_train=np.mat(Y_train).transpose()
Y_test=np.apply_along_axis(lambda x:0 if x=='M' else 1, 1,Y_test)
#Y_test=np.mat(Y_test).transpose()
#print(Y_train)
#print(Y_train.shape==a)
#https://blog.csdn.net/kingzone_2008/article/details/81067036
logR = LogisticRegression(solver='liblinear').fit(X_train, Y_train)

pred=logR.predict(X_test)

# 查看系数
print('Coefficients: \n', logR.coef_)
# 查看均方误差
print("Mean squared error: %.2f"%mean_squared_error(Y_test, pred))
# 解释方差分数:1是完美的预测
print('Variance score: %.2f' %r2_score(Y_test, pred))

#图输出
#plt.scatter(Y_test, pred, color='blue', linewidth=3)
#X=np.linspace(-100,100,10)
#plt.plot(X,X,color='green')
#plt.xticks(())
#plt.yticks(())
#plt.show()

print('降维（2）后:')
#print(newX)
newY=df.iloc[:,1:2].values


#print(df.iloc[:,1:2])
#print(newY)
newdata=np.concatenate((newY,newX),axis=1).transpose()
#origdata=pd.concat([df.iloc[:,1:1],newX])
#print(origdata)
#print(newdata)
#col_rand_array = np.arange(newdata.shape[1])
##np.random.shuffle(col_rand_array)
#col_rand = newdata[:,col_rand_array[0:500]]
#train=col_rand
train=newdata[:,:-50]
#col_rand = origdata[:,col_rand_array[0:100]]
#test=col_rand
test=newdata[:,-50:]
#print(train,'emmm',test)
#print('train:',train)
X_train=np.delete(train, 0, axis=0).transpose()
Y_train=train[[0]].transpose()
X_test=np.delete(test, 0, axis=0).transpose()
Y_test=test[[0]].transpose()
#a=Y_train.shape
Y_train=np.apply_along_axis(lambda x:0 if x=='M' else 1, 1,Y_train)
#print(Y_train.type)
#Y_train.apply()
#Y_train=np.mat(Y_train).transpose()
Y_test=np.apply_along_axis(lambda x:0 if x=='M' else 1, 1,Y_test)
#Y_test=np.mat(Y_test).transpose()
#print(Y_train)
#print(Y_train.shape==a)
#https://blog.csdn.net/kingzone_2008/article/details/81067036
logR = LogisticRegression(solver='liblinear').fit(X_train, Y_train)

pred=logR.predict(X_test)

# 查看系数
print('Coefficients: \n', logR.coef_)
# 查看均方误差
print("Mean squared error: %.2f"%mean_squared_error(Y_test, pred))
# 解释方差分数:1是完美的预测
print('Variance score: %.2f' %r2_score(Y_test, pred))

#图输出
plt.scatter(Y_test, pred, color='blue', linewidth=3)
#X=np.linspace(-100,100,10)
#plt.plot(X,X,color='green')
#plt.xticks(())
#plt.yticks(())
plt.show()
