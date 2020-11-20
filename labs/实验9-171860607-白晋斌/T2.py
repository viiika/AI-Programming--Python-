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

from sklearn import preprocessing

def trans(x,arg):
    return arg+'_'+x

def init(df):
    #df['uying'].apply(trans,args=("uying",))
    lst=['buying','maint','doors','persons','lug_boot','safety','class']
    for col in range(7):
        #print(col)
        #df[lst]=df[lst].apply(trans,args=(col,))
        df[lst[col]]=df[lst[col]].apply(trans,args=(lst[col],))
    #for col in ('buying','paint','doors','persons','lug_boot','safety','class'):

     #   df[col].apply(trans,args=(col,))
    #print(df)
    return df

def deal(data):
	return data.dropna().tolist()

def title1():
    df = pd.read_csv('car.data.csv')
    #print(df)
    #df=df.iloc[:,:-1]
    df = init(df)
    #df['buying']=df['buying'].apply(trans, args=("buying",))

    #print(df)
    df_arr = df.apply(deal, axis=1).tolist()
    te = TransactionEncoder()
    #ref:https://blog.csdn.net/qq_36523839/article/details/83960195
    te_ary = te.fit_transform(df_arr)
    res = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(res, min_support=0.2, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)  # 频繁项集可以按支持度排序
    print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= 2])  # 选择长度 >=2 的频繁项集

    association_rule = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)  # metric可以有很多的度量选项，返回的表列名都可以作为参数
    association_rule.sort_values(by='leverage', ascending=False, inplace=True)  # 关联规则可以按leverage排序

    print(association_rule)

def myfunc(k):
    if k=='vhigh':
        return 10
    if k=='high':
        return 8
    if k=='med':
        return 5
    if k=='low':
        return 2
    if k=='5more':
        return 6
    if k=='more':
        return 6
    if k=='small':
        return 2
    if k=='big':
        return 8
    return int(k)




def title2():
    df = pd.read_csv('car.data.csv')
    print(df)
    df.set_index(['class'], inplace=True)
    df = df.applymap(myfunc)
    print(df)

    res = df.groupby('class').mean()
    res = res.T
    print(res)
    # pic1

    plt.figure()
    res.plot()
    plt.show()

def title3():
    df = pd.read_csv('car.data.csv')
    df.set_index(['class'], inplace=True)
    df = df.applymap(myfunc)
    print(df)

    #做标准化处理和不做标准化处理的对比(df - df.mean()) / df.std()
    #df = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    #df=df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  #方法一
    ##
    print(df)
    kmodel = KMeans(n_clusters=4, n_jobs=8,n_init=200)  # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(df)  # 训练模型
    # ref:https://www.cnblogs.com/pinard/p/6169370.html
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(df.columns) + [u'类别数目']  # 重命名表头
    print(r)
    r = r.iloc[:, :-1]
    r = r.T
    # print(r)
    plt.figure()
    r.plot()
    plt.show()

if __name__ == '__main__':
    title3()



