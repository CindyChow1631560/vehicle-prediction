# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:26:32 2018

@author: asus
"""

import pandas as pd
import numpy as np 
import re
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.linear_model import ElasticNet,Lasso,SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import lightgbm as lgb


train_data0 = pd.read_csv('../data/yancheng_train_20171226.csv')
test_data = pd.read_csv('../data/yancheng_testA_20171225.csv')
test_data.rename(columns={'predict_date':'sale_date','predict_quantity':'sale_quantity'},inplace=True)
#
train_data = train_data0.loc[train_data0['sale_date']>201607]
#train_data=train_data0;
#for j in range(20157):
#    if train_data.loc[j,'sale_date'].astype('str')[4:6]=='11'or train_data.loc[j,'sale_date'].astype('str')[4:6]== '10'\
#    or train_data.loc[j,'sale_date'].astype('str')[4:6]== '9':
#        pass
#    else:
#        train_data=train_data.drop(j)
        
        
#train_data=train_data.reset_index(drop=True)

data_all = pd.concat((train_data, test_data), axis=0)

#data_all preprocessing on abnormal values
data_all['level_id'] = data_all['level_id'].apply(lambda x: 6 if x=='-' else x).astype('int')
data_all['fuel_type_id'] = data_all['fuel_type_id'].apply(lambda x:1 if x=='-' else x).astype('int')
data_all['power'] = data_all['power'].apply(lambda x:81 if x=='81/70' else x).astype('float')
data_all['engine_torque'] = data_all['engine_torque'].apply(lambda x: 155 if x=='155/140'\
                               else 73 if x=='-' else x).astype('float')
data_all['rated_passenger'] = data_all['rated_passenger'].apply(lambda x: 5 \
                                                 if x=='4-5' else 7 if x=='5-7' or x=='6-7'\
                                                 else 8 if x=='5-8' or x=='6-8' or x=='7-8'\
                                                 else x).astype('int')
data_all['TR'] = data_all['TR'].apply(lambda x: 7 if x=='8;7' else 5 if x=='5;4' else x).astype(int)
data_all['gearbox_type']=data_all['gearbox_type'].apply(lambda x: 'AT' if x=='AT;DCT' else 'MT' \
          if x=='MT;AT' else x)

#13 categorical fields 
cate_list = ['brand_id','type_id','level_id','department_id','TR','gearbox_type',\
             'if_charging','driven_type_id','fuel_type_id','newenergy_type_id',\
             'emission_standards_id','if_MPV_id','if_luxurious_id']

feature_cate_temp = pd.get_dummies(data_all[cate_list],prefix=cate_list,columns=cate_list)

#category原始数据drop，concat one-hot之后的数据
feature_temp1 = pd.concat((data_all.drop(cate_list,axis=1),feature_cate_temp),axis=1)
#class_id按照3年总销量排序,将class_id进行映射成140-1的数（更精确考虑每年）
class_list_temp = data_all.groupby(['class_id'],as_index=False)['sale_quantity']\
                           .sum().sort_values(by='sale_quantity', ascending=False)
class_list = pd.DataFrame(class_list_temp.loc[:,'class_id'])
class_list['class_trans'] = np.arange(140,0,-1)
feature_temp2 = pd.merge(feature_temp1, class_list).drop('class_id',1)

#sale_date：提取年、月的信息，其中年按照delta方式取，2012为1,2013为2......
sale_date_temp = pd.DataFrame(data_all['sale_date'].unique()).sort_values(by=0)
sale_date_temp['year_delta'] = (sale_date_temp[0]/100).astype('int')-2011
sale_date_temp['month'] = sale_date_temp[0].astype('str').apply(lambda x: x[4:6]).astype('int')
sale_date_temp.rename(columns={0:'sale_date'},inplace=True)
feature_temp3 = pd.merge(feature_temp2, sale_date_temp).drop('sale_date',1) 
#del sale_date_temp, train_trans1

#15 numerical fields
num_list = ['compartment','displacement','price_level','power','cylinder_number',\
            'engine_torque','car_length','car_width','car_height','total_quality',\
            'equipment_quality','rated_passenger','wheelbase','front_track','rear_track']
#将从price_level提出low_bound, up_bound， mean
price_level_bounds=feature_temp3['price_level'].apply(lambda x: list(map(int,re.findall(r'\d+',x))))
list_low = []
list_high = []
list_center = []
for i in price_level_bounds.values:
   if len(i) == 2:
      list_low.append(i[0])
      list_high.append(i[1])
      list_center.append((i[0]+i[1])/2)
   else:
      list_low.append(i[0])
      list_high.append(i[0])
      list_center.append(i[0])
   
feature_temp3['price_level_low'] = list_low
feature_temp3['price_level_high'] = list_high
feature_temp3['price_level_center'] = list_center

# 车体积/1000000
feature_temp3['car_volume'] = feature_temp3['car_length']*feature_temp3['car_width']*\
                              feature_temp3['car_height']/1000000
#2 key_fields
key_list = ['sale_date','class_id']

# target_field 
target_list = ['sale_quantity']

#***********************************构造特征*********************************#
feature_grouped = feature_temp3.groupby(['class_trans','year_delta','month'])
feature_sum = feature_grouped.sum()
feature_mean = feature_grouped.mean()
feature_std = feature_grouped.std()
feature_std=feature_std.fillna(0)
feature_record_num = feature_grouped['sale_quantity'].nunique()
#特征命名
feature_sum.columns = ['sum_%s' %x for x in feature_sum.columns]
feature_mean.columns = ['mean_%s' %x for x in feature_mean.columns]
feature_std.columns = ['std_%s' %x for x in feature_std.columns]
feature_record_num.columns = 'record_num'


#选特征concat
features = pd.concat((feature_mean,feature_std,feature_sum,feature_record_num),axis=1)
#features['class']=features.index
#features['class_num']=features.loc[:,'class'].astype(str)[0:1]
features=features.reset_index(drop=False)

predict_set = features[(features['year_delta']==6)&(features['month']==11)]
train_set = features[~((features['year_delta']==6)&(features['month']==11))]

#train predict sets
train_y = train_set.loc[:,'sum_sale_quantity']
train_x = train_set.drop(list(train_set.filter(regex='sale_quantity')),axis=1)

predict_x = predict_set.drop(list(predict_set.filter(regex='sale_quantity')),axis=1)


#modelling
#parameters={
#             'feature_fraction': np.logspace(0,1,20)
#        }
#grid=GridSearchCV(estimator= lgb.LGBMRegressor(objective='regression',num_leaves=5,max_bin = 55, bagging_fraction = 0.8,\
#                                               bagging_freq = 5, feature_fraction = 1,n_estimators=800,\
#                                               feature_fraction_seed=6, bagging_seed=9,learning_rate=0.33598,\
#                                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),param_grid=parameters,scoring='neg_mean_squared_error',cv=10)
#grid.fit(train_x,train_y)
#print(grid.best_params_)
#print(grid.best_score_)

#
KRR =  KernelRidge(alpha=359.38136,kernel='polynomial',coef0=12.9155,degree=1.1288)
GBoost = GradientBoostingRegressor(n_estimators=460, learning_rate=0.193069,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=2, 
                                   loss='ls', random_state =5)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.0000001, random_state=0))
#svr=SVR(C=316.2278,degree=2,gamma='auto',kernel='linear',coef0=0.01)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.00001, random_state=1))
#sgd=SGDRegressor(loss='squared_loss',alpha=0.6105,l1_ratio=0.22758,penalty='elasticnet',learning_rate='optimal',random_state=2)
model_xgb = xgb.XGBRegressor(colsample_bytree=1,subsample=0.71268,learning_rate=0.0373,\
                             max_depth=3,min_child_weight=1.05,gamma=13.895,n_estimators=2200,\
                             silent=1,random_state =7,reg_alpha=0.37276, reg_lambda=1.63789,nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.33598, n_estimators=800,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 1,
                              feature_fraction_seed=6, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(train_x,train_y)
xgb_pred = model_xgb.predict(train_x)
print("\nXGB score: {:.4f}\n".format(rmsle(train_y,xgb_pred)))
lasso.fit(train_x,train_y)
lasso_pred=lasso.predict(train_x)
print("\nlasso score: {:.4f}\n".format(rmsle(train_y,lasso_pred)))
ENet.fit(train_x,train_y)
ENet_pred=ENet.predict(train_x)
print("\ENet score: {:.4f}\n".format(rmsle(train_y,ENet_pred)))
GBoost.fit(train_x,train_y)
GBoost_pred=GBoost.predict(train_x)
print("\GBoost score: {:.4f}\n".format(rmsle(train_y,GBoost_pred)))
KRR.fit(train_x,train_y)
KRR_pred=KRR.predict(train_x)
print("\KRR score: {:.4f}\n".format(rmsle(train_y,KRR_pred)))
model_lgb.fit(train_x,train_y)
lgb_pred=model_lgb.predict(train_x)
print("\lgb score: {:.4f}\n".format(rmsle(train_y,lgb_pred)))
#sgd.fit(train_x,train_y)
#sgd_pred=sgd.predict(train_x)
#print("\nsgd score: {:.4f}\n".format(rmsle(train_y,sgd_pred)))
#svr.fit(train_x,train_y)
#svr_pred=svr.predict(train_x)
#print("\nsvr score: {:.4f}\n".format(rmsle(train_y,svr_pred)))
'''
Stacking models
'''
#Simplest Stacking approach : Averaging base models
#We begin with this simple approach of averaging base models. We build a new class 
#to extend scikit-learn with our model and also to laverage encapsulation and code reuse 
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
averaged_models.fit(train_x,train_y)
averaged_pred=averaged_models.predict(train_x)
print(" Averaged base models score: {:.4f}\n".format(rmsle(train_y,averaged_pred)))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
stacked_averaged_models.fit(train_x.values,train_y.values)
stack_pred=stacked_averaged_models.predict(train_x.values)
print("Stacking Averaged models score: {:.4f} \n".format(rmsle(train_y,stack_pred)))
print(rmsle(train_y,(stack_pred*0.70 +xgb_pred*0.15 + lgb_pred*0.15) ))


stacked_test_pred=stacked_averaged_models.predict(predict_x)
lgb_test_pred=model_lgb.predict(predict_x)
xgb_test_pred = model_xgb.predict(predict_x)
ensemble=stacked_test_pred*0.7+xgb_test_pred*0.15+lgb_test_pred*0.15

#ensemble1=pd.DataFrame(ensemble,np.arange(1,141))
#ensemble2 = ensemble1.reset_index()
#ensemble2.rename(columns={'index':'class_trans',0:'predict_quantity'},inplace=True)
#ensemble3 = ensemble2.merge(class_list,on='class_trans')
#ensemble3.drop('class_trans',axis=1,inplace=True)
#sub  = test_data.loc[:,['sale_date', 'class_id']]
#sub.rename(columns={'sale_date':'predict_date'},inplace=True)
#sub1 = sub.merge(ensemble3, on='class_id')
#
#sub1.to_csv('submission4.csv',index=False)


#xgb1 = pd.DataFrame(xgb_test_pred,np.arange(1,141))
#xgb2 = xgb1.reset_index()
#xgb2.rename(columns={'index':'class_trans',0:'predict_quantity'},inplace=True)
#xgb3 = xgb2.merge(class_list,on='class_trans')
#xgb3.drop('class_trans',axis=1,inplace=True)
#
#sub  = test_data.loc[:,['sale_date', 'class_id']]
#sub.rename(columns={'sale_date':'predict_date'},inplace=True)
#sub1 = sub.merge(xgb3, on='class_id')
#
#sub1.to_csv('submission4.csv',index=False)

lgb1 = pd.DataFrame(lgb_test_pred,np.arange(1,141))
lgb2 = lgb1.reset_index()
lgb2.rename(columns={'index':'class_trans',0:'predict_quantity'},inplace=True)
lgb3 = lgb2.merge(class_list,on='class_trans')
lgb3.drop('class_trans',axis=1,inplace=True)

sub  = test_data.loc[:,['sale_date', 'class_id']]
sub.rename(columns={'sale_date':'predict_date'},inplace=True)
sub1 = sub.merge(lgb3, on='class_id')

sub1.to_csv('submission_new1.csv',index=False)
