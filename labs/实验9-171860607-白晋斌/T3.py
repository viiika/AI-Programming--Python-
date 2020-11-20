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

#from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


threshold = 1.5 #离散点阈值
k = 2 #聚类的类别

def myfunc(k):
    if k=='n':
        return -1
    if k=='y':
        return 1
    if k=='?':
        return 0


def fun1():
    df = pd.read_csv('house-votes-84.csv', header=None)
    df.set_index([0], inplace=True)
    df = df.applymap(myfunc)

    index=[]
    for i in range(435):
        index.append(i)
    df1=df
    df1['index']=index
    df1.set_index(['index'],inplace=True)

    df1 = 1.0 * (df1 - df1.mean()) / df1.std()  # 数据标准化
    print(df1)


    kmodel = KMeans(n_clusters=k, n_jobs=4)  # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(df1)  # 训练模型
    # ref:https://www.cnblogs.com/pinard/p/6169370.html
    """
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(df.columns) + [u'类别数目']  # 重命名表头
    print(r)
    """
    #r = r.iloc[:, :-1]
    #r = r.T
    # print(r)
    #plt.figure()
    #r.plot()
    #plt.show()

    r = pd.concat([df1, pd.Series(kmodel.labels_, index=df.index)], axis=1)  # 每个样本对应的类别
    r.columns = list(df.columns) + [u'聚类类别']  # 重命名表头

    print(r)

    norm = []
    print(r.columns)
    for i in range(k):  # 逐一处理
        norm_tmp = r[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]][r[u'聚类类别'] == i] - kmodel.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)  # 求出绝对距离
        norm.append(norm_tmp / norm_tmp.median())  # 求相对距离并添加

    norm = pd.concat(norm)  # 合并
    print(norm)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    print("norm:")
    norm[norm <= threshold].plot(style='go')  # 正常点
    print(norm[norm <= threshold])

    discrete_points = norm[norm > threshold]  # 离群点
    print("discrete_points:")
    print(discrete_points)
    discrete_points.plot(style='ro')

    for i in range(len(discrete_points)):  # 离群点做标记
        id = discrete_points.index[i]
        n = discrete_points.iloc[i]
        plt.annotate('(%s, %0.2f)' % (id, n), xy=(id, n), xytext=(id, n))


    plt.xlabel(u'No.')
    plt.ylabel(u'relative distance')
    plt.show()




if __name__ == '__main__':
    fun1()