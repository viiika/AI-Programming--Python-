#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:55:16 2018

@author: dazhuang
"""

import os
import os.path
import json
import pandas as pd

path = 'ndb/ndb/ndb_total/'
files = os.listdir(path)

lst_name = []
lst_nutrients = []

for file in files :
     file_name = path + file
     data = json.load(open(file_name))
     name = data['report']['food']['name']
     lst_name.append(name)
     df = pd.DataFrame(data['report']['food']['nutrients'])
     del df['measures']
     lst_nutrients.append(df)

d = {}
k = 0
for food in lst_nutrients:
    x = float(food[food.name == 'Energy'].value)
    d[lst_name[k]] = x
    k += 1
    
last20_Energy = sorted(d.items(), key = lambda d: d[1], reverse=True)[-20:]
last20_Energy_df = pd.DataFrame(last20_Energy)
last20_Energy_df = last20_Energy_df.set_index(0, inplace=False, drop=True)

chart = last20_Energy_df.plot(kind = 'bar',legend = False)
chart.set_title("The Last 20")
chart.set_xlabel("Food")
chart.set_ylabel("Energy")