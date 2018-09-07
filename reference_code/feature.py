# -*- coding: utf-8 -*-
"""
@author: zixuanweeei

"""

import pandas as pd
import numpy as np
import re


def period_diff(x, n=1):
    if x.nunique() == 1:
        return np.array([-1])
    else:
        return np.diff(np.sort(x.unique()), n) / np.timedelta64(1, 'D')


def feature_gen(data, nonz_columns, hospitals):
    """Feature engineering

    Parameter
    --------
    data : `pandas.DataFrame` or `numpy.ndarray`
        data to be processed
    """
    data['交易月份'] = data['交易时间'].dt.month
    data_group = data[nonz_columns].groupby('个人编码')

    # == 医院种类
    features1 = data_group.agg({'医院编码': 'nunique'})\
                          .rename(columns={'医院编码': 'hospital_kinds'})
    # == 原始记录的总和、均值、方差
    features2 = data_group.agg(['sum', 'mean', 'std'])
    features2 = features2.drop('医院编码', axis=1, level=0)
    features2.columns = ['%s_%s' % (col, ope)
                         for col, ope in features2.columns]
    feature = pd.merge(features1, features2, left_index=True, right_index=True)
    del features1, features2

    # == 药品、检查、治疗、手术费用的相关比例(按记录)
    feature['药品费申报比例'] = feature['药品费申报金额_sum'] / feature['药品费发生金额_sum']
    feature['中草药费申报比例'] = feature['中草药费发生金额_sum'] / feature['药品费发生金额_sum']
    feature['中成药费申报比例'] = feature['中成药费发生金额_sum'] / feature['药品费发生金额_sum']
    feature['贵重药品申报比例'] = feature['贵重药品发生金额_sum'] / feature['药品费发生金额_sum']
    feature['药品自费比例'] = feature['药品费自费金额_sum'] / feature['药品费发生金额_sum']
    # ------------------------------------------------------------------------
    feature['贵重检查费申报比例'] = feature['贵重检查费金额_sum'] / feature['检查费发生金额_sum']
    feature['检查费自费申报比例'] = feature['检查费自费金额_sum'] / feature['检查费发生金额_sum']
    feature['检查费申报比例'] = feature['检查费申报金额_sum'] / feature['检查费发生金额_sum']
    # ------------------------------------------------------------------------
    feature['治疗费申报比例'] = feature['治疗费申报金额_sum'] / feature['治疗费发生金额_sum']
    feature['治疗费自费比例'] = feature['治疗费自费金额_sum'] / feature['治疗费发生金额_sum']
    # ------------------------------------------------------------------------
    feature['药品费审批比例'] = feature['药品费申报金额_sum'] / feature['本次审批金额_sum']
    feature['检查费审批比例'] = feature['检查费申报金额_sum'] / feature['本次审批金额_sum']
    feature['治疗费审批比例'] = feature['治疗费申报金额_sum'] / feature['本次审批金额_sum']
    feature['中草药费审批比例'] = feature['中草药费发生金额_sum'] / feature['本次审批金额_sum']    
    feature['中成药费审批比例'] = feature['中成药费发生金额_sum'] / feature['本次审批金额_sum']
    # ------------------------------------------------------------------------
# =============================================================================
#     feature['手术费申报比例'] = feature['手术费申报金额_sum'] / feature['手术费发生金额_sum']
#     feature['手术费自费比例'] = feature['手术费自费金额_sum'] / feature['手术费发生金额_sum']
#     # ------------------------------------------------------------------------
#     feature['医用材料费自费比例'] = feature['医用材料费自费金额_sum'] / feature['医用材料发生金额_sum']
#     feature['高价材料发生金额比例'] = feature['高价材料发生金额_sum'] / feature['医用材料发生金额_sum']
# =============================================================================
    feature[np.isinf(feature)] = 2  # 上述分母为零，代表总项目无消费状态，设置为2
    #总消费及比例
    feature['总消费'] = feature.loc[:, ['药品费发生金额_sum', '检查费发生金额_sum', '治疗费发生金额_sum', '手术费发生金额_sum', '医用材料发生金额_sum', '其它发生金额_sum']].sum(axis=1)
# =============================================================================
#     feature['药品费发生金额_sumrate'] = feature['药品费发生金额_sum'] / feature['总消费']
#     feature['检查费发生金额_sumrate'] = feature['检查费发生金额_sum'] / feature['总消费']
#     feature['治疗费发生金额_sumrate'] = feature['治疗费发生金额_sum'] / feature['总消费']
#     feature['手术费发生金额_sumrate'] = feature['手术费发生金额_sum'] / feature['总消费']
#     feature['医用材料发生金额_sumrate'] = feature['医用材料发生金额_sum'] / feature['总消费']
#     feature['其它发生金额_sumrate'] = feature['其它发生金额_sum'] / feature['总消费']
# =============================================================================
    feature['自负金额总消费比例'] = feature['起付标准以上自负比例金额_sum'] / feature['总消费']
    feature['非账户支付总消费比例'] = feature['非账户支付金额_sum'] / feature['总消费']
    
    #账户收支费用
    feature['总支付金额_mean'] = feature['基本医疗保险统筹基金支付金额_mean']+ \
                                feature['基本医疗保险个人账户支付金额_mean'] + feature['非账户支付金额_mean']
    feature['非统筹金额支付率_mean'] = (feature['基本医疗保险个人账户支付金额_mean'] + \
                                       feature['非账户支付金额_mean']) / feature['总支付金额_mean'] 
    feature['总支付金额_std'] = feature['基本医疗保险统筹基金支付金额_std']+ \
                               feature['基本医疗保险个人账户支付金额_std'] + feature['非账户支付金额_std']

    # 按每天消费做统计特性
    fee_related_cols = ['个人编码', '药品费发生金额', '检查费发生金额', '治疗费发生金额', '其它发生金额',
                        '贵重药品发生金额', '中成药费发生金额',
                        '起付标准以上自负比例金额', '基本医疗保险统筹基金支付金额',
                        '可用账户报销金额', '非账户支付金额', '本次审批金额', '交易时间']
    features3 = data[fee_related_cols].groupby(['个人编码', '交易时间']).sum()
    day_records_num = features3.groupby(level=0).apply(lambda x: x.shape[0])
    day_records_num = pd.DataFrame({'day_records_num': day_records_num})
    features3 = features3.groupby(level=0).agg(['std', 'mean', 'max', 'median'])
    features3.columns = ['%s_%s_byday' % (col, ope) for col, ope in features3.columns]
    feature = pd.merge(feature, features3, left_index=True, right_index=True)
    feature = pd.merge(feature, day_records_num, left_index=True, right_index=True)
    del features3, day_records_num

    # == 消费 - 时间相关性
# =============================================================================
#     fee_perday = data[fee_related_cols].groupby(['个人编码', '交易时间']).sum()
#     fee_perday = fee_perday.reset_index(level=1)
#     fee_perday['days'] = fee_perday['交易时间'].groupby(level=0).apply(lambda x: (x - x.head(1)).dt.days)
#     fee_corr = fee_perday.groupby(level=0).corr()
#     fee_corr = fee_corr.groupby(level=0).tail(1).iloc[:, :-1]
#     fee_corr.columns = ["%s_corrwithday" % col for col in fee_corr.columns]
#     fee_corr = fee_corr.reset_index(level=1, drop=True)
#     feature = pd.merge(feature, fee_corr, left_index=True, right_index=True)
#     del fee_perday, fee_corr
# =============================================================================
    

    # == 病种数，粗略统计
    diseases = data['出院诊断病种名称'].fillna(0).map(lambda x: len(re.split('\W+', x)) if x else 0)
    diseases = pd.DataFrame({'个人编码': data['个人编码'],
                             'diseases_kinds': diseases})
    diseases = diseases.groupby('个人编码').sum()
    feature = pd.merge(feature, diseases, left_index=True, right_index=True)
    del diseases

    # == 门诊特殊 & 挂号次数（按日）
    dis_reg = data[['个人编码', '交易时间', '出院诊断病种名称']].dropna()
    dis_reg['special'] = dis_reg['出院诊断病种名称'].map(lambda x: '门特' in x)
    dis_reg['registration'] = dis_reg['出院诊断病种名称'].map(lambda x: ('挂号' in x) and ('门特' not in x))
    dis_reg_perday = dis_reg.groupby(['个人编码', '交易时间']).sum()
    dis_reg_perday = dis_reg_perday.groupby(level=0).agg(['max', 'median', 'mean', 'std'])
    dis_reg_perday.columns = ['%s_%s_bymonth' % (k, l) for k, l in dis_reg_perday.columns]
    feature = pd.concat([feature, dis_reg_perday], axis=1).fillna(0)
    del dis_reg, dis_reg_perday

    # == 交易平均周期
    feature4 = data_group['交易时间'].apply(lambda x: period_diff(x).mean())
    feature4 = pd.DataFrame({'交易平均周期': feature4})
    feature = pd.merge(feature, feature4, left_index=True, right_index=True)
    del feature4

    # == 交易间隔方差
    feature5 = data_group['交易时间'].apply(lambda x: period_diff(x).std())
    feature5 = pd.DataFrame({'交易间隔方差': feature5})
    feature = pd.merge(feature, feature5, left_index=True, right_index=True)
    del feature5

    # == 是否去过存在诈骗记录的医院以及次数
    feature6_kinds = data_group['医院编码'].apply(lambda x: np.in1d(x.unique(), list(hospitals.keys())).sum())
    feature6_records = data_group['医院编码'].apply(lambda x: np.in1d(x.values, list(hospitals.keys())).sum())
    feature6_weights = data_group['医院编码'].apply(lambda x: np.sum([hospitals.get(k, 0) for k in x.unique()]))
    feature6 = pd.DataFrame({'hospital_cheat_kinds': feature6_kinds,
                             'hospital_cheat_records': feature6_records,
                             'hospital_cheat_weights': feature6_weights})
    feature = pd.merge(feature, feature6, left_index=True, right_index=True)
    del feature6, feature6_kinds, feature6_records, feature6_weights

    # == df_tain.csv 中的记录数
    records_num = data_group.apply(lambda x: x.shape[0])
    records_num = pd.DataFrame({'records_num': records_num})
    feature = pd.merge(feature, records_num, left_index=True, right_index=True)
    del records_num

    # 按月消费做统计特性
    fee_related_cols.append('交易月份')
    features7 = data[fee_related_cols].groupby(['个人编码', '交易月份']).sum()
    month_records_num = features7.groupby(level=0).apply(lambda x: x.shape[0])
    month_records_num = pd.DataFrame({'month_records_num': month_records_num})
    features7 = features7.groupby(level=0).agg(['std', 'mean', 'max', 'median'])
    features7.columns = ['%s_%s_bymonth' % (col, ope)
                         for col, ope in features7.columns]
    feature = pd.merge(feature, features7, left_index=True, right_index=True)
    feature = pd.merge(feature, month_records_num, left_index=True, right_index=True)
    del features7, month_records_num

    # == 门诊特殊 & 挂号次数（按月）
    dis_reg = data[['个人编码', '交易月份', '出院诊断病种名称']].dropna()
    dis_reg['special'] = dis_reg['出院诊断病种名称'].map(lambda x: '门特' in x)
    dis_reg['registration'] = dis_reg['出院诊断病种名称'].map(lambda x: '挂号' in x and ('门特' not in x))
    dis_reg_permonth = dis_reg.groupby(['个人编码', '交易月份']).sum()
    dis_reg_permonth = dis_reg_permonth.groupby(level=0).agg(['max', 'median', 'mean', 'std'])
    dis_reg_permonth.columns = ['%s_%s_bymonth' % (k, l) for k, l in dis_reg_permonth.columns]
    feature = pd.concat([feature, dis_reg_permonth], axis=1).fillna(0)
    del dis_reg, dis_reg_permonth

    return feature


def fee_feature_gen(fee):
    fee['药剂总价'] = fee['单价']*fee['数量']
    fee['费用发生月份'] = fee['费用发生时间'].dt.month

    # 日发生消费的医院数的最大值
    group = fee.groupby(['个人编码', '费用发生时间'])
    hospitals_perday = group['医院编码'].apply(lambda x: x.nunique())
    hospitals_perday = hospitals_perday.groupby(level=0).agg(['max', 'median'])
    hospitals_perday.columns = ['hospitals_perday_%s' % k for k in hospitals_perday.columns]

    # 日均药剂花费以及数量
    drug_fee_perday = group['药剂总价'].sum()
    drug_fee_perday = drug_fee_perday.groupby(level=0).agg(['max', 'mean', 'std'])
    drug_fee_perday.columns = ['drug_fee_perday_%s' % k for k in drug_fee_perday.columns]
    drug_counts_perday = group['数量'].sum()
    drug_counts_perday = drug_counts_perday.groupby(level=0).agg(['max', 'mean'])
    drug_counts_perday.columns = ['drug_counts_perday_%s' % k for k in drug_counts_perday.columns]    
    drug_perday = pd.merge(drug_fee_perday, drug_counts_perday, left_index=True, right_index=True)
    feature = pd.merge(hospitals_perday, drug_perday, left_index=True, right_index=True)
    del drug_counts_perday, drug_fee_perday, drug_perday, hospitals_perday

    # 药剂 - 时间相关性
    drug_corr = group[['数量', '药剂总价']].sum()
    drug_corr = drug_corr.reset_index(level=1)
    drug_corr['days'] = drug_corr['费用发生时间'].groupby(level=0).apply(lambda x: (x - x.head(1)).dt.days)
    drug_corr = drug_corr.groupby(level=0).corr()
    drug_corr = drug_corr.groupby(level=0).tail(1).iloc[:, :-1]
    drug_corr.columns = ['%s_corrwithday' % k for k in drug_corr.columns]
    drug_corr = drug_corr.reset_index(level=1, drop=True)
    feature = pd.merge(feature, drug_corr, left_index=True, right_index=True)
    del drug_corr

    # 日均三目统计项目
    cata3_perday = group['三目统计项目'].sum()
    cata3_perday = cata3_perday.groupby(level=0).agg(['max', 'mean', 'std'])
    cata3_perday.columns = ['cata3_perday_%s' % k for k in cata3_perday.columns]
    feature = pd.merge(feature, cata3_perday, left_index=True, right_index=True)

    # 三目统计项目 - 时间相关性
    cata3_corr = group['三目统计项目'].sum()
    cata3_corr = cata3_corr.reset_index(level=1)
    cata3_corr['days'] = cata3_corr['费用发生时间'].groupby(level=0).apply(lambda x: (x - x.head(1)).dt.days)
    cata3_corr = cata3_corr.groupby(level=0).corr()
    cata3_corr = cata3_corr.groupby(level=0).tail(1).iloc[:, :-1]
    cata3_corr = cata3_corr.reset_index(level=1, drop=True)
    cata3_corr.columns = ['三目统计项目_corrwithday']
    feature = pd.merge(feature, cata3_corr, left_index=True, right_index=True)
    del cata3_corr

    # 月均更换医院次数(+1)  
    group = fee.groupby(['个人编码', '费用发生月份'])
    hospitals_permonth = group['医院编码'].apply(lambda x: x.nunique())
    hospitals_permonth = hospitals_permonth.groupby(level=0).agg(['max', 'mean', 'std'])
    hospitals_permonth.columns = ['hospitals_permonth_%s' % k for k in hospitals_permonth.columns]

    feature = pd.merge(feature, hospitals_permonth, left_index=True, right_index=True)
    del hospitals_permonth

    # 月均药剂花费以及数量
    drug_fee_permonth = group['药剂总价'].sum()
    drug_fee_permonth = drug_fee_permonth.groupby(level=0).agg(['max', 'mean', 'std'])
    drug_fee_permonth.columns = ['drug_fee_permonth_%s' % k for k in drug_fee_permonth.columns]
    drug_counts_permonth = group['数量'].sum()
    drug_counts_permonth = drug_counts_permonth.groupby(level=0).agg(['max', 'mean', 'std'])
    drug_counts_permonth.columns = ['drug_counts_permonth_%s' % k for k in drug_counts_permonth.columns]
    drug_permonth = pd.merge(drug_fee_permonth, drug_counts_permonth, left_index=True, right_index=True)
    feature = pd.merge(feature, drug_permonth, left_index=True, right_index=True)

    # 月均三目统计项目
    cata3_permonth = group['三目统计项目'].sum()
    cata3_permonth = cata3_permonth.groupby(level=0).agg(['max', 'mean', 'std'])
    cata3_permonth.columns = ['cata3_permonth_%s' % k for k in cata3_permonth.columns]
    feature = pd.merge(feature, cata3_permonth, left_index=True, right_index=True)
    
    # 剂型种类

    med_group = fee[['个人编码', '剂型']].dropna().groupby('个人编码')
    feature['med_kinds_total'] = med_group.count()
    feature['med_kinds_nunique'] = pd.DataFrame(med_group['剂型'].nunique()) #每个人有多少种剂型
    feature['med_kinds_rep_rate'] = (feature['med_kinds_total'] - feature['med_kinds_nunique']) / feature['med_kinds_total']
    
# =============================================================================
#     med_kind_int = {label:idx for idx, label in enumerate(set(fee['剂型']))}
#     med_group1 = pd.concat([fee['个人编码'],fee['剂型'].map(med_kind_int)],axis=1).groupby('个人编码')
#     feature['med_kinds_unique'] = pd.DataFrame(med_group1['剂型'].unique())
# =============================================================================
# =============================================================================
#     feature = pd.merge(feature, med_kinds_total, left_index=True, right_index=True)
#     feature = pd.merge(feature, med_kinds_nunique, left_index=True, right_index=True)
#     feature = pd.merge(feature, med_kinds_unique, left_index=True, right_index=True)
#     feature = pd.merge(feature, med_kinds_rep_rate, left_index=True, right_index=True)
# =============================================================================
    return feature
