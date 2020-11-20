import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


df=pd.read_csv('dataset.csv',header=None)
df=df.iloc[:,:-1]
argv=df.iloc[:,1:]
#低方差滤波（Low Variance Filter）
#如果我们有一个数据集，其中某列的数值基本一致，也就是它的方差非常低，那么这个变量还有价值吗？和上一种方法的思路一致，我们通常认为低方差变量携带的信息量也很少，所以可以把它直接删除。
#放到实践中，就是先计算所有变量的方差大小，然后删去其中最小的几个。需要注意的一点是：方差与数据范围相关的，因此在采用该方法前需要对数据做归一化处理。
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(argv)
x_minmax=pd.DataFrame(x_minmax)
#print(x_minmax.var())
maxvar=x_minmax.var().sort_values(ascending=False)
print(maxvar)
x_minmax=x_minmax.iloc[:,[0,4]]
res = pd.concat([df.iloc[:,[0]],x_minmax],axis=1,ignore_index=True)
#print(x_minmax)
print(res)
#区别不大，该方案作废

#随机森林（Random Forest）
#随机森林是一种广泛使用的特征选择算法，它会自动计算各个特征的重要性，所以无需单独编程。这有助于我们选择较小的特征子集。
#在开始降维前，我们先把数据转换成数字格式，因为随机森林只接受数字输入。同时，ID这个变量虽然是数字，但它目前并不重要，所以可以删去。

"""
RF=argv.drop(0, axis=0)
model = RandomForestRegressor(random_state=1, max_depth=10)
RF=pd.get_dummies(RF)

model.fit(RF,argv.y_增长率)

features = RF.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:9])  # 因指标太多，选取前10个指标作为例子
plt.title('Index selection')
plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative importance of indicators')
plt.show()
"""
#print(df)

#独立成分分析法（ICA）是基于信息理论的一种常用的降维方法，其与PCA主要的不同是PCA是寻找不相关的变量，而ICA是挑选独立变量。同时PCA主要对于高斯分布数据比较有效，而ICA适用于其他分布。我们可以调用sklearn中的FastICA函数来进行数据独立成分分析。

from sklearn.decomposition import FastICA
ICA = FastICA(n_components=2, random_state=12)
X=ICA.fit_transform(argv)
print(X)
X=X.transpose()
#plt.scatter(X[0],X[1])
#plt.show()


#PCA
from sklearn.decomposition import PCA
model = PCA(n_components=2)   #降到2维
model.fit(argv)                  #训练
newX=model.fit_transform(argv)   #降维后的数据  ###transform就会执行降维操作
print(model.explained_variance_ratio_)  #输出贡献率
#print(newX)                  #输出降维后的数据
#print(newX)

#newX=newX.transpose()
pltfig=df.iloc[:,:1]
#print('pltfig:',pltfig)
pd2=pd.DataFrame(newX)
#print('pd2:')
#print(pd2)
pltfig=pd.concat([pltfig,pd2],ignore_index=True,axis=1)
print(pltfig)
#pltfig=pltfig.groupby(pltfig[0])
#pltfig.plot('scatter')


colors = {'A':'red', 'B':'blue', 'C':'green','D':'yellow'}
fig, ax = plt.subplots()
grouped = pltfig.groupby(0)
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x=1, y=2, label=key, color=colors[key])
    #print(key)
plt.show()

#scatter(pltfig.groupby(0), x=1, y=2, alpha=0.5)
#ddd=pltfig.groupby(0)
#ddd.plot.scatter(x=1,y=2)
#for name,group in pltfig:
#        plt.plot(group.area,group.poptotal,marker='o',linestyle='',ms=10,label=name)
#plt.show()
#newXX=pd.DataFrame(newX)
#print(pltfig)
#plt.scatter(newX[0],newX[1])
#plt.show()

#降成三维尝试绘制三维散点图

#PCA
model3 = PCA(n_components=3)   #降到2维
model3.fit(argv)                  #训练
newX3=model3.fit_transform(argv)   #降维后的数据  ###transform就会执行降维操作
print(model3.explained_variance_ratio_)  #输出贡献率

pltfig=df.iloc[:,:1]
pd3=pd.DataFrame(newX3)
#print('pd2:')
#print(pd2)
pltfig=pd.concat([pltfig,pd3],ignore_index=True,axis=1)
print(pltfig)
#pltfig=pltfig.groupby(pltfig[0])
#pltfig.plot('scatter')
    #plt.scatter(,projection='3d')
#Axes3D.scatter(x=1,y=2,z=3,s=10)
Axes3D.scatter3D(pltfig.)
Axes3D.imshow()

colors = {'A':'red', 'B':'blue', 'C':'green','D':'yellow'}
fig, ax = plt.subplots()
grouped = pltfig.groupby(0)
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x=1, y=2,h=3, label=key, color=colors[key],projection = '3d')
    #print(key)
plt.show()


#以下是自行对数据进行分类部分
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取用于聚类的数据，并创建数据表
df=pd.DataFrame(pd.read_csv('dataset.csv',header=0))
df=df.iloc[:,:-1]
argv=df.iloc[:,1:]


from sklearn.decomposition import PCA
model = PCA(n_components=2)   #降到2维
model.fit(argv)                  #训练
newX=model.fit_transform(argv)   #降维后的数据  ###transform就会执行降维操作

loan_data=pd.DataFrame(newX)
#设置要进行聚类的字段
loan = np.array(loan_data[[0,1]])
print(loan)
#设置类别为3
clf=KMeans(n_clusters=4)
clf=clf.fit(loan)
loan_data['label']=clf.labels_#标记
#将数据代入到聚类模型中

print(loan_data)

colors = {0:'red', 1:'blue', 2:'green',3:'yellow'}
fig, ax = plt.subplots()
grouped = loan_data.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x=0, y=1, label=key, color=colors[key])
    #print(key)
plt.show()




"""
