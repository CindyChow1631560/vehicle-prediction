# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:03:07 2018

@author: asus
"""
#==============================================================================
# import os  
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'  
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] 
# from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
# from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
#==============================================================================
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import re
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn.feature_selection import RFECV
'''
使用excel转化后的数据
'''
train_data = pd.read_csv('../data/yancheng_train_20171226.csv')
test_data = pd.read_csv('../data/yancheng_testA_20171225.csv')

for j in range(20157):
    if train_data.loc[j,'sale_date'].astype('str')[4:6]=='11'or train_data.loc[j,'sale_date'].astype('str')[4:6]== '10'\
    or train_data.loc[j,'sale_date'].astype('str')[4:6]== '9':
        pass
    else:
        train_data=train_data.drop(j)
        
        
train_data=train_data.reset_index(drop=True)
for j in range(3191):
    if train_data.loc[j,'fuel_type_id']==2\
    or train_data.loc[j,'fuel_type_id']==3\
    or train_data.loc[j,'newenergy_type_id']==2\
    or train_data.loc[j,'newenergy_type_id']==4\
    or train_data.loc[j,'emission_standards_id']==3\
    or train_data.loc[j,'emission_standards_id']==5:
        train_data=train_data.drop(j)
    else:
        pass
#train_datus=np.where(train_data['sale_date']>201612,1,0)
#for j in range(20157):
#    if train_datus[j]!=1:
#        train_data=train_data.drop(j)
#    else:
#        pass
#查看数据信息
train_data.info()
#level_id.isnull().all() = True,只有class_id=178529的有问题,不妨level_id=6
#fuel_type_id,class_id=175962,961962; 同类型的车fti都是1，就补1
#.......
train_data['level_id'] = train_data['level_id'].apply(lambda x: 6 if x=='-' else x).astype('int')
train_data['fuel_type_id'] = train_data['fuel_type_id'].apply(lambda x:1 if x=='-' else x).astype('int')
train_data['power'] = train_data['power'].apply(lambda x:81 if x=='81/70' else x).astype('float')
train_data['engine_torque'] = train_data['engine_torque'].apply(lambda x: 155 if x=='155/140'\
                               else 73 if x=='-' else x).astype('float')
train_data['rated_passenger'] = train_data['rated_passenger'].apply(lambda x: 5 \
                                                 if x=='4-5' else 7 if x=='5-7' or x=='6-7'\
                                                 else 8 if x=='5-8' or x=='6-8' or x=='7-8'\
                                                 else x).astype('int')
train_data['TR'] = train_data['TR'].apply(lambda x: 7 if x=='8;7' else 5 if x=='5;4' else x).astype(int)
train_data['gearbox_type']=train_data['gearbox_type'].apply(lambda x: 'AT' if x=='AT;DCT' else 'MT' \
          if x=='MT;AT' else x)

#------------------------拟合正态分布---------------------------------#
#==============================================================================
# (mu, sigma) = norm.fit(train_data['sale_quantity'])#拟合正态分布
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SaleQuantity distribution')
# 
# fig = plt.figure()
# res = stats.probplot(train_data['sale_quantity'], plot=plt)
# plt.show()
# 
# train_data["sale_quantity"] = np.log1p(train_data["sale_quantity"])
# #Check the new distribution 
# sns.distplot(train_data['sale_quantity'] , fit=norm);
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train_data['sale_quantity'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SaleQuantity distribution')
# 
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train_data['sale_quantity'], plot=plt)
# plt.show()
# 
# #----------------------------拟合正态分布----------------------------------#
# numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
# 
# # Check the skew of all numerical features
# skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(10)
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# 
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     train_data[feat] = boxcox1p(train_data[feat], lam)
# #----------------------------------------------------------------------------#
# #-----------------------------------------------------------------------------------#
# 
#==============================================================================

#13 categorical fields 
cate_list = ['brand_id','type_id','level_id','department_id','TR','gearbox_type',\
             'if_charging','driven_type_id','fuel_type_id','newenergy_type_id',\
             'emission_standards_id','if_MPV_id','if_luxurious_id']

feature_cate_temp = pd.get_dummies(train_data[cate_list],prefix=cate_list,columns=cate_list)

#---------------------------构造特征准备数据------------------------------------#
#category原始数据drop，concat one-hot之后的数据
feature_temp1 = pd.concat((train_data.drop(cate_list,axis=1),feature_cate_temp),axis=1)
#class_id按照3年总销量排序,将class_id进行映射成140-1的数（更精确考虑每年）
class_list_temp = train_data.groupby(['class_id'],as_index=False)['sale_quantity']\
                           .sum().sort_values(by='sale_quantity', ascending=False)
class_list = pd.DataFrame(class_list_temp.loc[:,'class_id'])
class_list['class_trans'] = np.arange(140,0,-1)
feature_temp2 = pd.merge(feature_temp1, class_list).drop('class_id',1)
#del class_list_temp

#sale_date：提取年、月的信息，其中年按照delta方式取，2012为1,2013为2......
sale_date_temp = pd.DataFrame(train_data['sale_date'].unique()).sort_values(by=0)
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
features = pd.concat((feature_mean,feature_std,feature_sum),axis=1)
#features['class']=features.index
#features['class_num']=features.loc[:,'class'].astype(str)[0:1]
features=features.reset_index(drop=False)
out_y=feature_sum['sum_sale_quantity']


#out_y=out_y.reset_index(drop=False)

#最后要把 class_trans, year_delta, month 从标签里取出来作为特征

#test数据映射
#test_trans = pd.merge(test_data,class_list,on='class_id')
#test_trans['year_delta'] = [6]*140
#test_trans['month'] = [11]*140
#test = test_trans[['class_trans','year_delta','month']]
#其它：还可按照销量构造某特征在各个取值的重要性特征
#      构造一条记录里某特征出现了几个取值特征
'''
processing testing data
'''
test_data['level_id'] = test_data['level_id'].apply(lambda x: 6 if x=='-' else x).astype('int')
test_data['fuel_type_id'] = test_data['fuel_type_id'].apply(lambda x:1 if x=='-' else x).astype('int')
test_data['power'] = test_data['power'].apply(lambda x:81 if x=='81/70' else x).astype('float')
test_data['engine_torque'] = test_data['engine_torque'].apply(lambda x: 155 if x=='155/140'\
                               else 73 if x=='-' else x).astype('float')
test_data['rated_passenger'] = test_data['rated_passenger'].apply(lambda x: 5 \
                                                 if x=='4-5' else 7 if x=='5-7' or x=='6-7'\
                                                 else 8 if x=='5-8' or x=='6-8' or x=='7-8'\
                                                 else x).astype('int')

#-------------------------拟合正态分布---------------------------------------------#
#==============================================================================
# numeric_feats = test_data.dtypes[test_data.dtypes != "object"].index
# 
# # Check the skew of all numerical features
# skewed_feats = test_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(10)
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# 
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     test_data[feat] = boxcox1p(test_data[feat], lam)
#==============================================================================
#-----------------------------------------------------------------------------------#


feature_cate_temp_test = pd.get_dummies(test_data[cate_list],prefix=cate_list,columns=cate_list)
#---------------------------构造特征准备数据------------------------------------#
#category原始数据drop，concat one-hot之后的数据
feature_temp1_test = pd.concat((test_data.drop(cate_list,axis=1),feature_cate_temp_test),axis=1)
#class_id按照3年总销量排序,将class_id进行映射成140-1的数（更精确考虑每年）
class_list_temp_test = test_data.groupby(['class_id'],as_index=False)['predict_quantity']\
                           .sum().sort_values(by='predict_quantity', ascending=False)
class_list_test = pd.DataFrame(class_list_temp_test.loc[:,'class_id'])
class_list_test['class_trans'] = np.arange(140,0,-1)
feature_temp2_test = pd.merge(feature_temp1_test, class_list_test).drop('class_id',1)
#del class_list_temp

#sale_date：提取年、月的信息，其中年按照delta方式取，2012为1,2013为2......
sale_date_temp_test = pd.DataFrame(test_data['predict_date'].unique()).sort_values(by=0)
sale_date_temp_test['year_delta'] = (sale_date_temp_test[0]/100).astype('int')-2011
sale_date_temp_test['month'] = sale_date_temp_test[0].astype('str').apply(lambda x: x[4:6]).astype('int')
sale_date_temp_test.rename(columns={0:'predict_date'},inplace=True)
feature_temp3_test = pd.merge(feature_temp2_test, sale_date_temp_test).drop('predict_date',1) 
price_level_bounds_test=feature_temp3_test['price_level'].apply(lambda x: list(map(int,re.findall(r'\d+',x))))
list_low_test = []
list_high_test = []
list_center_test = []
for i in price_level_bounds_test.values:
   if len(i) == 2:
      list_low_test.append(i[0])
      list_high_test.append(i[1])
      list_center_test.append((i[0]+i[1])/2)
   else:
      list_low_test.append(i[0])
      list_high_test.append(i[0])
      list_center_test.append(i[0])
   
feature_temp3_test['price_level_low'] = list_low_test
feature_temp3_test['price_level_high'] = list_high_test
feature_temp3_test['price_level_center'] = list_center_test

# 车体积/1000000
feature_temp3_test['car_volume'] = feature_temp3_test['car_length']*feature_temp3_test['car_width']*\
                              feature_temp3_test['car_height']/1000000


#***********************************构造特征*********************************#
feature_grouped_test = feature_temp3_test.groupby(['class_trans','year_delta','month'])
feature_sum_test = feature_grouped_test.sum()
feature_mean_test = feature_grouped_test.mean()
feature_std_test = feature_grouped_test.std()
feature_std_test=feature_std_test.fillna(0)
feature_record_num_test = feature_grouped_test['predict_quantity'].nunique()
#特征命名
feature_sum_test.columns = ['sum_%s' %x for x in feature_sum_test.columns]
feature_mean_test.columns = ['mean_%s' %x for x in feature_mean_test.columns]
feature_std_test.columns = ['std_%s' %x for x in feature_std_test.columns]
feature_record_num_test.columns = 'record_num'


#选特征concat
features_test = pd.concat((feature_mean_test,feature_std_test,feature_sum_test),axis=1)
features_test=features_test.reset_index(drop=False)




'''
modeling
'''
#n_folds = 5
##cv = StratifiedShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0)
#def rmsle_cv(model):
#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(features.values)
#    rmse= np.sqrt(-cross_val_score(model, features.values, out_y, scoring="neg_mean_squared_error", cv = kf))
#    return(rmse)
##adjust parameters of models
#parameters={
#             'learning_rate': np.logspace(-4,0,20),
#             'n_estimators': [2000,2100,2200,2300,2400,2500]
#        }
#grid=GridSearchCV(estimator= xgb.XGBRegressor(colsample_bytree=1,subsample=0.71268,max_depth=3,min_child_weight=1.05,gamma=13.895,\
#                  silent=1,random_state =7,reg_alpha=0.37276, reg_lambda=1.63789,nthread = -1),param_grid=parameters,scoring='neg_mean_squared_error',cv=10)
#grid.fit(features,out_y)
#print(grid.best_params_)
#print(grid.best_score_)

#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
###KRR = KernelRidge(alpha=0.6, kernel='sigmoid', gamma=1, coef0=2.5)
#RFR = RandomForestRegressor(n_estimators=600)
#GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                   max_depth=4, max_features='sqrt',
#                                   min_samples_leaf=15, min_samples_split=10, 
#                                   loss='ls', random_state =5)
#model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                             learning_rate=0.05, max_depth=3, 
#                             min_child_weight=1.7817, n_estimators=2200,
#                             reg_alpha=0.4640, reg_lambda=0.8571,
#                             subsample=0.5213, silent=1,
#                             random_state =7, nthread = -1)

#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                              learning_rate=0.05, n_estimators=720,
#                              max_bin = 55, bagging_fraction = 0.8,
#                              bagging_freq = 5, feature_fraction = 0.2319,
#                              feature_fraction_seed=9, bagging_seed=9,
#                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#score = rmsle_cv(lasso)
#print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#score = rmsle_cv(ENet)
#print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#score = rmsle_cv(RFR)
#print("RandomForest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#score = rmsle_cv(GBoost)
#print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=1,subsample=0.71268,learning_rate=0.0373,\
                             max_depth=3,min_child_weight=1.05,gamma=13.895,n_estimators=2200,\
                             silent=1,random_state =7,reg_alpha=0.37276, reg_lambda=1.63789,nthread = -1)
#score = rmsle_cv(model_xgb)
#print(score)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(features.values,out_y)
xgb_train_pred=model_xgb.predict(features.values)
print(rmsle(out_y,xgb_train_pred))
xgb_pred=model_xgb.predict(features_test.values)
sub = pd.DataFrame()
sub['predict_date'] = test_data['predict_date']
sub['class_id']=class_list_test['class_id']
sub['predict_quantity'] = xgb_pred
sub.to_csv('submission_new.csv',index=False)
#score = rmsle_cv(model_lgb)
#print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
#
#'''
#Stacking models
#'''
##Simplest Stacking approach : Averaging base models
##We begin with this simple approach of averaging base models. We build a new class 
##to extend scikit-learn with our model and also to laverage encapsulation and code reuse 
#class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#    def __init__(self, models):
#        self.models = models
#        
#    # we define clones of the original models to fit the data in
#    def fit(self, X, y):
#        self.models_ = [clone(x) for x in self.models]
#        
#        # Train cloned base models
#        for model in self.models_:
#            model.fit(X, y)
#
#        return self
#    
#    #Now we do the predictions for cloned models and average them
#    def predict(self, X):
#        predictions = np.column_stack([
#            model.predict(X) for model in self.models_
#        ])
#        return np.mean(predictions, axis=1)   
#    
#averaged_models = AveragingModels(models = (ENet, GBoost, RFR, lasso))
#
#score = rmsle_cv(averaged_models)
#print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
#    def __init__(self, base_models, meta_model, n_folds=5):
#        self.base_models = base_models
#        self.meta_model = meta_model
#        self.n_folds = n_folds
#   
#    # We again fit the data on clones of the original models
#    def fit(self, X, y):
#        self.base_models_ = [list() for x in self.base_models]
#        self.meta_model_ = clone(self.meta_model)
#        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
#        
#        # Train cloned base models then create out-of-fold predictions
#        # that are needed to train the cloned meta-model
#        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#        for i, model in enumerate(self.base_models):
#            for train_index, holdout_index in kfold.split(X, y):
#                instance = clone(model)
#                self.base_models_[i].append(instance)
#                instance.fit(X[train_index], y[train_index])
#                y_pred = instance.predict(X[holdout_index])
#                out_of_fold_predictions[holdout_index, i] = y_pred
#                
#        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
#        self.meta_model_.fit(out_of_fold_predictions, y)
#        return self
#   
#    #Do the predictions of all base models on the test data and use the averaged predictions as 
#    #meta-features for the final prediction which is done by the meta-model
#    def predict(self, X):
#        meta_features = np.column_stack([
#            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
#            for base_models in self.base_models_ ])
#        return self.meta_model_.predict(meta_features)
#    
##To make the two approaches comparable (by using the same number of models) , 
##we just average Enet KRR and Gboost, then we add lasso as meta-model.
#    
#stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, RFR),
#                                                 meta_model = lasso)
#
#score = rmsle_cv(stacked_averaged_models)
#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#
#'''
#Ensembling StackedRegressor, XGBoost and LightGBM
#'''
#def rmsle(y, y_pred):
#    return np.sqrt(mean_squared_error(y, y_pred))
#
##stackedregressor
#stacked_averaged_models.fit(features.values, out_y)
#stacked_train_pred = stacked_averaged_models.predict(features.values)
#stacked_pred = np.expm1(stacked_averaged_models.predict(features_test.values))
#print(rmsle(out_y, stacked_train_pred))
#
##XGBoosr
#model_xgb.fit(features.values, out_y)
#xgb_train_pred = model_xgb.predict(features.values)
#xgb_pred = np.expm1(model_xgb.predict(features_test.values))
#print(rmsle(out_y, xgb_train_pred))
##LightGBM
#model_lgb.fit(features, out_y)
#lgb_train_pred = model_lgb.predict(features_test.values)
#lgb_pred = np.expm1(model_lgb.predict(features_test.values))
#print(rmsle(out_y, lgb_train_pred))
#print('RMSLE score on train data:')
#print(rmsle(out_y,stacked_train_pred*0.70 +
#               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
#ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
