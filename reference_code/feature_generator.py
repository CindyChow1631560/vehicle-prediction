"""
@author: zixuanweeei

Feature enginnering
"""

import pandas as pd
import numpy as np
from feature import feature_gen, fee_feature_gen

# check nonzero columns(features) in train data
# only the nonzero columns(features) are preserved to be processed
train_data = pd.read_csv('../data/df_train.csv', header=0)
test_data = pd.read_csv('../data/df_test.csv', header=0)
label = pd.read_csv('../data/df_id_train.csv', header=None)
label.columns = ['个人编码', 'label']
fee = pd.read_csv('../data/fee_detail.csv')

foreign_key = pd.concat([train_data[['顺序号', '个人编码']], test_data[['顺序号', '个人编码']]])
fee = pd.merge(foreign_key, fee, on='顺序号')
fee['费用发生时间'] = pd.to_datetime(fee['费用发生时间'], format='%Y-%m-%d', errors='coerce')
fee_feature = fee_feature_gen(fee)
del fee

# 统计出现诈骗情况的医院
train_data_ = pd.merge(train_data, label, on='个人编码')
hospitals = train_data_.loc[train_data_['label'] == 1, :]
hospitals = hospitals.groupby('个人编码')
hospitals = hospitals['医院编码'].unique()
hospitals = np.concatenate(hospitals.values)
hospitals, counts = np.unique(hospitals, return_counts=True)
hospitals = dict(zip(hospitals, counts))
del train_data_

train_data = train_data.drop(train_data.loc[:, '手术费自费金额':'手术费申报金额'].columns, axis=1)
test_data = test_data.drop(test_data.loc[:, '手术费自费金额':'手术费申报金额'].columns, axis=1)
train_data = train_data.drop(train_data.loc[:, '床位费拒付金额':'床位费申报金额'].columns, axis=1)
test_data = test_data.drop(test_data.loc[:, '床位费拒付金额':'床位费申报金额'].columns, axis=1)
train_data = train_data.drop(train_data.loc[:, '医用材料费拒付金额':'成分输血申报金额'].columns, axis=1)
test_data = test_data.drop(test_data.loc[:, '医用材料费拒付金额':'成分输血申报金额'].columns, axis=1)
train_data = train_data.drop(train_data.loc[:, '其它拒付金额':'起付线标准金额'].columns, axis=1)
test_data = test_data.drop(test_data.loc[:, '其它拒付金额':'起付线标准金额'].columns, axis=1)
train_data = train_data.drop(['医疗救助个人按比例负担金额', '最高限额以上金额'], axis=1)
test_data = test_data.drop(['医疗救助个人按比例负担金额', '最高限额以上金额'], axis=1)
train_data = train_data.drop(['公务员医疗补助基金支付金额', '城乡救助补助金额'], axis=1)
test_data = test_data.drop(['公务员医疗补助基金支付金额', '城乡救助补助金额'], axis=1)
train_data = train_data.drop(train_data.loc[:, '补助审批金额':'家床起付线剩余'].columns, axis=1)
test_data = test_data.drop(test_data.loc[:, '补助审批金额':'家床起付线剩余'].columns, axis=1)

train_data['交易时间'] = pd.to_datetime(train_data['交易时间'], format='%Y/%m/%d', errors='coerce')
test_data['交易时间'] = pd.to_datetime(test_data['交易时间'], format='%Y/%m/%d', errors='coerce')

train_data_ = train_data.fillna(0)
zeros_check_train = (train_data_ == 0).all()
zeros_columns_train = train_data_.columns[zeros_check_train]
nonz_columns_train = train_data_.columns[~zeros_check_train]

test_data_ = test_data.fillna(0)
zeros_check_test = (test_data_ == 0).all()
zeros_columns_test = test_data_.columns[zeros_check_test]
nonz_columns_test = test_data_.columns[~zeros_check_test]
del train_data_, test_data_

common_nonz_columns = nonz_columns_train.join(nonz_columns_test, how='inner')

train_feature = feature_gen(train_data, common_nonz_columns, hospitals)
train_feature = pd.merge(train_feature, fee_feature, left_index=True, right_index=True)
label = pd.read_csv('../data/df_id_train.csv', index_col=0, header=None)
label.columns = ['Label']
train_feature = pd.merge(train_feature, label, left_index=True, right_index=True)
train_feature.to_csv('../data/train_feature.csv')

test_feature = feature_gen(test_data, common_nonz_columns, hospitals)
test_feature = pd.merge(test_feature, fee_feature, left_index=True, right_index=True)
test_feature.to_csv('../data/test_feature.csv')
