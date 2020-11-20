import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def myfunc(k):
    if k=='n':
        return -1
    if k=='y':
        return 1
    if k=='?':
        return 0



def myjob1():
    df=pd.read_csv('house-votes-84.csv',header=None)
    df.set_index([0], inplace=True)
    df=df.applymap(myfunc)
    print(df)
    republican=df.loc["republican",:]
    democrat=df.loc["democrat",:]
    #print(democrat.mean())
    res=df.groupby(0).mean()
    nres=df.groupby(0).var()
    res=res.T
    nres=nres.T
    #print(df.groupby(0).mean())
    print(res)
    print(nres)
    #pic1

    plt.figure()
    res.plot()
    #plt.show()
    return res

def myjob2():
    df = pd.read_csv('house-votes-84.csv', header=None)
    df.set_index([0], inplace=True)
    df = df.applymap(myfunc)
    kmodel = KMeans(n_clusters=2, n_jobs=4,n_init=5)  # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(df)  # 训练模型
    #ref:https://www.cnblogs.com/pinard/p/6169370.html
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(df.columns) + [u'类别数目']  # 重命名表头
    print(r)
    r=r.iloc[:,:-1]
    r=r.T
    #print(r)
    plt.figure()
    r.plot()
    #plt.show()
    return r

if __name__ == '__main__':
    orig=myjob1()
    calu=myjob2()
    res=pd.DataFrame()
    res['0']=orig['democrat']-calu[0]

    res['1']=orig['republican']-calu[1]
    print(orig)
    print(calu)
    print(res)