# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:30:08 2018

@author: zhouxue
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6

train=pd.read_csv("../data/train.csv")
print(train.head())
print(train.dtypes)

dateparse=lambda sale_date: pd.datetime.strptime(sale_date,"%Y%m")
train=pd.read_csv("../data/train.csv")
train["sale_date"]=dateparse("sale_date")

