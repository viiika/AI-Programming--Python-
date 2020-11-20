import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('data_akbilgic.csv')
#print(df.shape)

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
#降维
X=np.array(df[['ISE(USD BASED)','SP(imkb_x)','DAX','FTSE','NIKKEI','BOVESPA','EU','EM']])

from sklearn.decomposition import PCA
model3 = PCA(n_components=7)   #降到2维
model3.fit(X)                  #训练
newX=model3.fit_transform(X)   #降维后的数据  ###transform就会执行降维操作
print(newX)

df2=pd.DataFrame(newX)
#df=df.concat(df[:,0:1],newX)
#df=df.concatenate((df[:,0:1],newX),axis=1)
df=pd.concat([df.iloc[:,[0,1]],df2],axis=1,ignore_index=True)

print(df)

#df.set_index(["date"], inplace=True)
#线性回归
#data=df.drop(columns='date')
data=df.values.transpose()

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

