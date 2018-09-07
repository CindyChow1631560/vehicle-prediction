# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:00:08 2018

@author: asus
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn.feature_selection import RFECV


color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test_data.csv')

#choose data which is in month 9-11 
train.loc[1,'sale_date'].astype('str')[0:6]
for j in range(20157):
    if train.loc[j,'sale_date'].astype('str')[4:6]=='11'or train.loc[j,'sale_date'].astype('str')[4:6]== '10'\
    or train.loc[j,'sale_date'].astype('str')[4:6]== '9':
        pass
    else:
        train=train.drop(j)
    
    
    
print("The train data size is : {} ".format(train.shape))
print("The test data size is : {} ".format(test.shape))
fig, ax = plt.subplots()
ax.scatter(x = train['type_id'], y = train['sale_quantity'])
plt.ylabel('sale_quantity', fontsize=13)
plt.xlabel('type_id', fontsize=13)
plt.show()

sns.distplot(train['sale_quantity'] , fit=norm);##distplot( )为hist加强版，kdeplot( )为密度曲线图

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['sale_quantity'])#拟合正态分布
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SaleQuantity distribution')

fig = plt.figure()
res = stats.probplot(train['sale_quantity'], plot=plt)
plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["sale_quantity"] = np.log1p(train["sale_quantity"])

#Check the new distribution 
sns.distplot(train['sale_quantity'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['sale_quantity'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SaleQuantity distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['sale_quantity'], plot=plt)
plt.show()

'''
Features engineering
'''
#let's first concatenate the train and test data in the same dataframe
ntrain =train.shape[0]
ntest = test.shape[0]
y_train= train.sale_quantity.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['sale_quantity'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#calculate missing data rate
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#analyze data correlation
#Correlation map to see how features are correlated with SaleQuantity
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
'''
inputing missing data
'''
all_data = all_data.drop('price',axis=1)

all_data['engine_torque']=all_data['engine_torque'].fillna(all_data['engine_torque'].mean())
all_data['rated_passenger']=all_data['rated_passenger'].fillna(all_data['rated_passenger'].mean())
all_data['level_id']=all_data['level_id'].fillna(all_data['level_id'].mean())
all_data['fuel_type_id']=all_data['fuel_type_id'].fillna(all_data['fuel_type_id'].mean())
all_data['volume']=all_data['car_length']*all_data['car_width']*all_data['car_height']
#Check remaining missing values if any
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


#Transforming some numerical variables that are really categorical
#all_data['sale_date']=all_data['sale_date'].astype(str)
#all_data['class_id']=all_data['class_id'].astype(str)
#all_data['brand_id']=all_data['brand_id'].astype(str)
#all_data['level_id']=all_data['level_id'].astype(str)
#all_data['type_id']=all_data['type_id'].astype(str)
#all_data['driven_type_id']=all_data['driven_type_id'].astype(str)
#all_data['fuel_type_id']=all_data['fuel_type_id'].astype(str)
#all_data['newenergy_type_id']=all_data['newenergy_type_id'].astype(str)
#all_data['emission_standards_id']=all_data['emission_standards_id'].astype(str)
#all_data['if_MPV_id']=all_data['if_MPV_id'].astype(str)
#all_data['if_luxurious_id']=all_data['if_luxurious_id'].astype(str)


cols=['sale_date','class_id','brand_id','compartment','type_id','level_id','department_id','TR',\
     'gearbox_type','displacement','if_charging','price_level','driven_type_id','fuel_type_id',\
     'newenergy_type_id','emission_standards_id','if_MPV_id','if_luxurious_id','power','cylinder_number',\
     'engine_torque','car_length','car_width','car_height','total_quality','equipment_quality',\
     'rated_passenger','wheelbase','front_track','rear_track','volume']
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#Getting dummy categorical features
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]

#'''
#feature selection
#'''
#def get_features(df, columns,predict, model=None):
#    newDf = df.copy()
#    newDf = newDf.select_dtypes(['number'])
#    newDf = newDf.dropna(axis=1, how='any')
#    
#    #dropColumns = ["PassengerId", "Survived"]
#    #newDf = newDf.drop(dropColumns, axis = 1)
#    
#    all_X = newDf[columns]
#    all_y = predict
#    
#    cv = StratifiedShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
#    if model == None:
#        classifier = tree.DecisionTreeClassifier(
#                criterion="entropy",
#                max_depth=10,
#                max_features='auto',
#                max_leaf_nodes=None,
#                min_samples_leaf = 10,
#                )
#    else:
#        classifier = model
#    selector = RFECV(classifier, scoring = 'roc_auc', cv=cv, step = 1)
#    selector.fit(all_X,all_y)
#    rfecv_columns = all_X.columns[selector.support_]
#    return rfecv_columns
#
#rfecv_features =get_features(all_data,cols,y_train,model=None)
#print(len(rfecv_features))
#print(rfecv_features)

'''
modeling
'''
import os  
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'  
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] 
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


#We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut,
# we add then one line of code, in order to shuffle the dataset prior to cross-validation
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
##KRR = KernelRidge(alpha=0.6, kernel='sigmoid', gamma=1, coef0=2.5)
RFR = RandomForestRegressor(n_estimators=600)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(RFR)
print("RandomForest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
core = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

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
    
averaged_models = AveragingModels(models = (ENet, GBoost, RFR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
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
    
#To make the two approaches comparable (by using the same number of models) , 
#we just average Enet KRR and Gboost, then we add lasso as meta-model.
    
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, RFR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

'''
Ensembling StackedRegressor, XGBoost and LightGBM
'''
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#stackedregressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

#XGBoosr
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

#use the result of Lasso regression
lasso.fit(train,y_train)
lasso_pred=np.expm1(lasso.predict(test))

sub = pd.DataFrame()
sub['predict_date'] = test['sale_date']
sub['class_id']=test['class_id']
sub['sale_quantity'] = ensemble
sub.to_csv('submission4.csv',index=False)