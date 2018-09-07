# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:26:51 2018

@author: asus
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

test_list_temp=train.groupby(['class_id'],as_index=False).apply(lambda x: x.mode())
           # .sort_values(by='sale_quantity', ascending=False)    
test_data=pd.merge(test,test_list_temp,on="class_id").drop(['sale_date','sale_quantity'],1)

sub = pd.DataFrame(test_data)
sub.to_csv('test_data.csv',index=False)