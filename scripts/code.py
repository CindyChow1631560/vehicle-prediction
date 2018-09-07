# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:51:12 2018

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit,GridSearchCV,StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression,SGDRegressor
train=pd.read_csv('../data/train.csv')
test=pd.read_csv('../data/test.csv')

del train["gearbox_type"]
del train["if_charging"]
del train["price_level"]
#train0=train.drop(train["sale_quantity"])
train["level_id"]=train["level_id"].fillna(0)
train["engine_torque"]=train["engine_torque"].fillna(0)
train["rated_passenger"]=train["rated_passenger"].fillna(0)
train["fuel_type_id"]=train["fuel_type_id"].fillna(0)

train["sale_date"]=pd.to_datetime(train["sale_date"],format='%Y%m',errors='coerce')
test["predict_date"]=pd.to_datetime(test["predict_date"],format='%Y%m',errors='coerce')
train["volume"]=train["car_length"]*train["car_height"]*train["car_width"]

test_data=pd.merge(train,test,how='inner',on="class_id").groupby("class_id").mean()


del test_data["sale_quantity"]


grouped=train.groupby(["class_id"])
group_mean=grouped.mean()
group_sum=grouped.sum()
group_std=grouped.std()
feature1=group_mean.drop("sale_quantity",1)
feature2=group_sum.drop("sale_quantity",1)
feature3=group_std.drop("sale_quantity",1)
#train=pd.merge(train,feature1)
#train=pd.merge(train,feature2)
#train=pd.merge(train,feature3)
'''
test_data=test_data.rename(columns={"predict_date":"sale_date"})
test_data=test_data.rename(columns={"predict_quantity":"sale_quantity"})
test_group=test_data.groupby([test_data["class_id"],test_data["sale_date"]])
group_test1=test_group.mean()
group_test2=test_group.sum()
group_test3=test_group.std()
test_feature1=group_test1.drop("sale_quantity",1)
test_feature2=group_test2.drop("sale_quantity",1)
test_feature3=group_test3.drop("sale_quantity",1)
#test_data=pd.merge(test_data,feature_test1)
#test_data=pd.merge(test_data,feature_test2)
#test_data=pd.merge(test_data,feature_test3)
'''

#train["price_level"].unique()
#train["price_level"]=train["price_level"].map({'8-10W':3,'10-15W':4,'5WL':1,'15-20W':5,'5-8W':2,'20-25W':6,
#     '25-35W':7,'35-50W':8,'50-75W':9,'NaN':0})

'''
column=["brand_id","compartment","type_id","level_id","department_id","TR","gearbox_type","displacement","if_charging",
          "price_level","driven_type_id","fuel_type_id","newenergy_type_id","emission_standards_id","if_MPV_id","if_luxurious_id",
          "power","cylinder_number","engine_torque","car_length","car_width","car_height","total_quality","equipment_quality",
          "rated_passenger","wheelbase","front_track","rear_track"]
'''
#column=["class_id","sale_date"]

#for columns in column:
#    train_dummies=pd.get_dummies(train[columns],prefix=columns)
#    train=pd.concat([train,train_dummies],axis=1)
'''
test.head()    
col=["class_id","predict_date"]

for c in col:
    test_dummies=pd.get_dummies(test[c],prefix=c)
    test=pd.concat([test,test_dummies],axis=1)

train.head(2)
'''
#模型选择
def get_model(df,features):
    train_x=features
    train_y=df["sale_quantity"]
    cv=ShuffleSplit(n_splits=10,train_size=0.7,test_size=0.3,random_state=0)
    model_param=[
        {
            "name": "AdaBoostRegressor",
            "estimator": AdaBoostRegressor(random_state=0),
            "hyperparameters":
            {
                "n_estimators": [20,50,80,110],
                "learning_rate":  np.logspace(-9, 3, 13)
            }
        },
        {
            "name": "RandomForestRegressor",
            "estimator": SGDRegressor(random_state=0),
            "hyperparameters":
            {
                "penalty": ["l2","l1","elasticnet"],
                "alpha": np.logspace(-9, 3, 13),
                "learning_rate": ["constant","optimal","invscaling"]
            }
        }
    ]
    models=[]
    for model in model_param:
        grid=GridSearchCV(estimator=model["estimator"],param_grid=model["hyperparameters"],cv=10)
        grid.fit(train_x,train_y)
    
        model_att={
            "model": grid.best_estimator_,
            "best_param": grid.best_params_,
            "best_score": grid.best_score_,
            "grid":grid    
        }
        models.append(model_att)
        print("model and its parameters:")
        print(grid.best_params_)
        print(grid.best_score_)
    return models

## feature selection using RFECV
def get_features(df,features,model=None):
    newDf=df.copy()
    newDf = newDf.select_dtypes(['number'])
    newDf = newDf.dropna(axis=1, how='any')
    all_X = newDf[features]
    all_y = df["sale_quantity"]
    cv=StratifiedShuffleSplit(n_splits=10,train_size=.7,test_size=.3,random_state=0)
    if model==None:
        regressor=AdaBoostRegressor(n_estimators=100)
    else:
        regressor=model
    selector = RFECV(regressor, scoring = 'roc_auc', cv=cv, step = 1)
    selector.fit(all_X,all_y) 
    rfecv_columns = all_X.columns[selector.support_]
    return rfecv_columns


models = get_model(train,feature1)

#select the best one based on its index from console
best_grid=models[0]["grid"]
best_regressor=models[0]["model"]
best_param=models[0]["best_param"]

predictions=best_regressor.predict(test_data.drop("predict_quantity",1))

sub={"predict_date": test["predict_date"],"class_id": test["class_id"],"predict_quantity": predictions}
submission=pd.DataFrame(sub)
submission.to_csv(path_or_buf="Submission.csv", index=False, header=True)