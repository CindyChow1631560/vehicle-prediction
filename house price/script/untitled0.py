# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:39:46 2018

@author: asus
"""

v_len =int(input())
edge=[[0]*5]*5
print(edge)
dist1={'A':0,'B':1,'C':2,'D':3,'E':4,}
for i in range(v_len):
    strList=input()
    for j in range(0,len(strList)-2,2):
        first=dist1[strList[j]]
        second=dist1[strList[j+2]]
        edge[first][second]=int(strList[j+1])
print(edge)