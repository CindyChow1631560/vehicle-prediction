# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:53:22 2018

@author: LIU
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit,GridSearchCV
'''
使用excel转化后的数据
'''
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


#将训练数据中各class_id的11月份的值提取出来

#train_data_temp=train_data
'''
train_data_temp.loc[1,'sale_date'].astype('str')[4:6]
for j in range(20157):
    if train_data_temp.loc[j,'sale_date'].astype('str')[4:6]!='11':
        train_data_temp=train_data_temp.drop(j)
    else:
        pass
 '''   
    
#将train_data_temp中的norminal类型的数据转换成数字
#train_data_temp['if_charging']=train_data_temp['if_charging'].apply(lambda x: 1 if x=='L' else 0)
#train_data_temp['price_level']=train_data_temp['price_level'].map({'8-10W':3,'10-15W':4,'5WL':1,'15-20W':5,'5-8W':2,'20-25W':6,
     #'25-35W':7,'35-50W':8,'50-75W':9})

   
test_list_temp=train_data_temp.groupby(['class_id'],as_index=False).apply(lambda x: x.mode())
           # .sort_values(by='sale_quantity', ascending=False)    
test_data=pd.merge(test_data,test_list_temp,on="class_id").drop(['sale_date'],1)
del test_list_temp

#class_id按照总销量排序,将class_id进行映射成140-1的数
class_list_temp = train_data.groupby(['class_id'],as_index=False).sum()\
            .sort_values(by='sale_quantity', ascending=False)
class_list = pd.DataFrame(class_list_temp.loc[:,'class_id'])
class_list['class_trans'] = np.arange(140,0,-1)
train_trans1 = pd.merge(train_data, class_list).drop('class_id',1)
del class_list_temp

#构造test中的class_id对应的数字
class_list_test = test_data.groupby(['class_id'],as_index=False).sum()\
            .sort_values(by='sale_quantity', ascending=False)
class_listt = pd.DataFrame(class_list_test.loc[:,'class_id'])
class_listt['class_trans'] = np.arange(140,0,-1)
test_trans1 = pd.merge(test_data, class_listt).drop('class_id',1)
del class_list_test


#sale_date：提取年、月的信息，其中年按照delta方式取，2012为1,2013为2......
sale_date_temp = pd.DataFrame(train_data['sale_date'].unique()).sort_values(by=0)
sale_date_temp['year_delta'] = (sale_date_temp[0]/100).astype('int')-2011
sale_date_temp['month'] = sale_date_temp[0].astype('str').apply(lambda x: x[4:6]).astype('int')
sale_date_temp.rename(columns={0:'sale_date'},inplace=True)
train_trans = pd.merge(train_trans1, sale_date_temp).drop('sale_date',1) 
del sale_date_temp, train_trans1


#predict_date
predict_date_temp = pd.DataFrame(test_data['predict_date'].unique()).sort_values(by=0)
predict_date_temp['year_delta'] = (predict_date_temp[0]/100).astype('int')-2011
predict_date_temp['month'] = predict_date_temp[0].astype('str').apply(lambda x: x[4:6]).astype('int')
predict_date_temp.rename(columns={0:'predict_date'},inplace=True)
test_trans = pd.merge(test_trans1, predict_date_temp).drop(['predict_date','sale_quantity'],1) 
del predict_date_temp, test_trans1

#填充NaN值
train_trans['price']=train_trans['price'].fillna(train_trans['price'].mean())
train_trans['engine_torque']=train_trans['engine_torque'].fillna(train_trans['engine_torque'].mean())
train_trans['rated_passenger']=train_trans['rated_passenger'].fillna(train_trans['rated_passenger'].mean())
train_trans['level_id']=train_trans['level_id'].fillna(0)
train_trans['fuel_type_id']=train_trans['fuel_type_id'].fillna(train_trans['fuel_type_id'].mode())
########
test_trans['price']=test_trans['price'].fillna(test_trans['price'].mean())
test_trans['engine_torque']=test_trans['engine_torque'].fillna(test_trans['engine_torque'].mean())
test_trans['rated_passenger']=test_trans['rated_passenger'].fillna(test_trans['rated_passenger'].mean())
test_trans['level_id']=test_trans['level_id'].fillna(0)
test_trans['fuel_type_id']=test_trans['fuel_type_id'].fillna(test_trans['fuel_type_id'].mode())


#if_charging:L-无增压：1，T-涡轮增压:0
if_charging_temp = train_trans['if_charging'].apply(lambda x: 1 if x=='L' else 0)\
                                .rename('if_charging_trans')
train_trans = pd.concat((train_trans,if_charging_temp),axis=1).drop('if_charging',1)
####
if_charging_test = test_trans['if_charging'].apply(lambda x: 1 if x=='L' else 0)\
                                .rename('if_charging_trans')
test_trans = pd.concat((test_trans,if_charging_test),axis=1).drop('if_charging',1)

#将价格区间映射成数字
price_level_temp=train_trans["price_level"].map({'8-10W':3,'10-15W':4,'5WL':1,'15-20W':5,'5-8W':2,'20-25W':6,
     '25-35W':7,'35-50W':8,'50-75W':9}).rename('price_level_trans')
train_trans=pd.concat([train_trans,price_level_temp],axis=1).drop('price_level',1)
####
price_level_test=test_trans["price_level"].map({'8-10W':3,'10-15W':4,'5WL':1,'15-20W':5,'5-8W':2,'20-25W':6,
     '25-35W':7,'35-50W':8,'50-75W':9}).rename('price_level_trans')
test_trans=pd.concat([test_trans,price_level_test],axis=1).drop('price_level',1)

#Norminal定类变量需要先按照class_id提取出来(假设每种class_id对应的这些)
norm_list = ['gearbox_type','if_charging','price_level']
train_trans_group = train_trans.groupby(['class_trans','year_delta','month'])
feature10 = train_trans_group.agg({'gearbox_type':'nunique'}).rename(columns={'gearbox_type':'gbt_kinds'})
feature11 = train_trans_group.agg({'gearbox_type':'count'}).rename(columns={'gearbox_type':'gbt_record_num'})
features1 = pd.concat((feature10,feature11),axis=1)

#测试数据
test_trans_group = test_trans.groupby(['class_trans','year_delta','month'])
test10 = test_trans_group.agg({'gearbox_type':'nunique'}).rename(columns={'gearbox_type':'gbt_kinds'})
test11 = test_trans_group.agg({'gearbox_type':'count'}).rename(columns={'gearbox_type':'gbt_record_num'})
test1 = pd.concat((test10,test11),axis=1)

feature20 = train_trans_group.agg({'if_charging_trans':'nunique'}).rename(columns={'if_charging_trans':'charge_kinds'})
feature21 = train_trans_group.agg({'if_charging_trans':'count'}).rename(columns={'if_charging_trans':'charge_record_num'})
feature22 = train_trans_group.agg({'if_charging_trans':'sum'}).rename(columns={'if_charging_trans':'charge_type'})
features2 = pd.concat((feature20,feature21,feature22),axis=1)

#测试数据
test20 = test_trans_group.agg({'if_charging_trans':'nunique'}).rename(columns={'if_charging_trans':'charge_kinds'})
test21 = test_trans_group.agg({'if_charging_trans':'count'}).rename(columns={'if_charging_trans':'charge_record_num'})
test22 = test_trans_group.agg({'if_charging_trans':'sum'}).rename(columns={'if_charging_trans':'charge_type'})
test2 = pd.concat((test20,test21,test22),axis=1)

feature30=train_trans_group.agg({'price_level_trans': 'nunique'}).rename(columns={'price_level_trans':'price_kinds'})
feature31=train_trans_group.agg({'price_level_trans': 'count'}).rename(columns={'price_level_trans': 'price_record_num'})
feature32=train_trans_group.agg({'price_level_trans': 'sum'}).rename(columns={'price_level_trans': 'price_type'})
features3=pd.concat((feature30,feature31,feature32),axis=1)

#测试数据
test30=test_trans_group.agg({'price_level_trans': 'nunique'}).rename(columns={'price_level_trans':'price_kinds'})
test31=test_trans_group.agg({'price_level_trans': 'count'}).rename(columns={'price_level_trans': 'price_record_num'})
test32=test_trans_group.agg({'price_level_trans': 'sum'}).rename(columns={'price_level_trans': 'price_type'})
test3=pd.concat((test30,test31,test32),axis=1)

del train_data["gearbox_type"]
del train_data["if_charging"]
del train_data["price_level"]
del test_data["gearbox_type"]
del test_data["if_charging"]
del test_data["price_level"]
#Numeric数值变量求均值、总和、标准差构造特征
features4 = train_trans_group.mean().drop('sale_quantity',1)
test4=test_trans_group.mean().drop('predict_quantity',1)
features=pd.concat([features1,features2,features3,features4],axis=1)
tests=pd.concat([test1,test2,test3,test4],axis=1)
out_y=train_trans_group.agg({'sale_quantity': 'mean'})


hyperparameters={'n_estimators': [10,20,30,40,50]}

cv=ShuffleSplit(n_splits=10,train_size=0.7,test_size=0.3,random_state=0)
grid=GridSearchCV(estimator=RandomForestRegressor(random_state =0),param_grid=hyperparameters,cv=10)
grid.fit(features,out_y)

best_regressor=grid.best_estimator_
predictions=best_regressor.predict(tests)

sub={"predict_date": test_data["predict_date"],"class_id": test_data["class_id"],"predict_quantity": predictions}
submission=pd.DataFrame(sub)
submission.to_csv(path_or_buf="Submission.csv", index=False, header=True)




