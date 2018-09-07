# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:29:37 2018

@author: asus
"""
import pandas as pd
import numpy as np 


sub1 = pd.read_csv('../scripts/submission4.csv')
sub2 = pd.read_csv('../data/yancheng_testB_20180224.csv')
sub=sub2.merge(sub1,on='class_id')

sub.to_csv('sub2.csv',index=False)