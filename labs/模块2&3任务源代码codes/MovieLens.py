import pandas as pd
import numpy as np

# 读入数据
unames = ['user id', 'age', 'gender', 'occupation', 'zip code']
users = pd.read_table('ml-100k/u.user', sep = '\|', names = unames, engine='python')
rnames = ['user id', 'item id', 'rating', 'timestamp']
ratings = pd.read_table('ml-100k/u.data', sep='\t', names = rnames, engine='python')

# 选择需要的数据列，提高效率
users_df = pd.DataFrame()
users_df['user id'] = users['user id']
users_df['gender'] = users['gender']
ratings_df = pd.DataFrame()
ratings_df['user id'] = ratings['user id']
ratings_df['rating'] = ratings['rating']

# 将数据合并
rating_df = pd.merge(users_df, ratings_df)
gender_table = pd.pivot_table(rating_df, index = ['gender', 'user id'], values = 'rating')
### 不考虑每个人自身评分的聚合 x  = pd.pivot_table(rating_df, index = ['gender'], values = 'rating', aggfunc = pd.Series.std)

# 利用pandas中的数据透视表pivot_table()函数对数据进行聚合，gender_table中的数据形式为：
# gender  user id
# F       2          3.709677
#         5          2.874286
# …
# M       898        3.500000
#         899        3.525926
# …
### gender_df.groupby('gender').apply(np.std)
gender_df = pd.DataFrame(gender_table)
gender_table.groupby('gender').apply(pd.Series.std)  # np.std 
'''
# 分男女，过滤透视表
Female_df = gender_df.query("gender == ['F']")
Male_df = gender_df.query("gender == ['M']")
# 按性别计算评分的标准差
Female_std = pd.Series.std(Female_df)
Male_std = pd.Series.std(Male_df)
print ('Gender', '\nF\t%.6f' % Female_std, '\nM\t%.6f' % Male_std)
'''