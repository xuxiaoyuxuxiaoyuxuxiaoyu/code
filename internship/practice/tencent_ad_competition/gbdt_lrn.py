import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

import argparse
import numpy as np
from data_ import partdata
from memory_profiler import  profile
files=['adFeature.csv','userFeature.csv','train.csv','test1.csv']
import gc
# @profile(precision=4)#显示内存使用情况
def data_preprocess(args):
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    data, aid_index = partdata()
    data_dic = {}
    for aid in aid_index:
        data_dic[aid] = data.loc[data['aid'] == aid]
        for feature in one_hot_feature:
            try:
                data_dic[aid][feature] = LabelEncoder().fit_transform(data_dic[aid][feature].apply(int))  # 对训练集的one-hot特征进行编码，0，1，2，3.。。
            except:
                data_dic[aid][feature] = LabelEncoder().fit_transform(data_dic[aid][feature])
    train_dic = {}
    test_dic = {}
    train_label_dic = {}
    res_dic = {}
    for key in data_dic.keys():
        train = data_dic[key][data_dic[key].label != -1]
        train_y = train.pop('label')
        test = data_dic[key][data_dic[key].label == -1]
        res = test[['aid', 'uid']]
        test = test.drop('label', axis=1)
        enc = OneHotEncoder()
        train_x = train[['creativeSize']]
        test_x = test[['creativeSize']]

        for feature in one_hot_feature:
            enc.fit(data_dic[key][feature].values.reshape(-1, 1))
            train_a = enc.transform(train[feature].values.reshape(-1, 1))
            test_a = enc.transform(test[feature].values.reshape(-1, 1))
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('one-hot prepared !')
        cv = CountVectorizer()
        for feature in vector_feature:
            cv.fit(data[feature])
            train_a = cv.transform(train[feature])
            test_a = cv.transform(test[feature])
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('cv prepared !')
        train_dic[key] = train_x
        train_label_dic[key] = train_y
        test_dic[key] = test_x
        res_dic[key] = res

    # train = data[data.label != -1]  # 提取训练集
    # train_y = train[['label']]  # 获取标签
    # # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    # test = data[data.label == -1]  # 提取测试集
    # res_train = train[['aid', 'uid']]
    # res = test[['aid', 'uid']]
    # test = test.drop('label', axis=1)
    # enc = OneHotEncoder()
    # train_x = train[['aid']]
    # test_x = test[['aid']]
    #
    # for feature in one_hot_feature:
    #     enc.fit(data[feature].values.reshape(-1, 1))
    #     train_a = enc.transform(train[feature].values.reshape(-1, 1))
    #     test_a = enc.transform(test[feature].values.reshape(-1, 1))
    #     train_x = sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    # print('one-hot prepared !')
    #
    # cv = CountVectorizer()
    # for feature in vector_feature:
    #     cv.fit(data[feature])
    #     train_a = cv.transform(train[feature])
    #     test_a = cv.transform(test[feature])
    #     train_x = sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    # print('cv prepared !')
    # train_x = sparse.hstack((train_x,train_y))
    return train_dic,train_label_dic,test_dic,res_dic

def patchextract(train_dic,train_label_dic,test_dic,res_dic):
    # train = pd.concat([train_x,train_y],axis=1)
    model_tree={}
    submission = pd.DataFrame([])
    for aid in train_dic.keys():
        train_X, test_X, train_y, test_y = train_test_split(train_dic[aid],
                                                            train_label_dic[aid],
                                                            test_size=0.2,
                                                            random_state=1)
        train_single_label = train_y
        train_single_data = train_X
        model_tree[aid] = lgb.LGBMClassifier(
        boosting_type='gbdt',min_child_samples=10, num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=8000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=10, random_state=2018, n_jobs=100
        )
        model_tree[aid].fit(
        train_single_data, train_single_label,
        eval_set=[(test_X,test_y)], eval_metric='auc'#, early_stopping_rounds=1000
        )
        a=0
        for i in model_tree[aid].feature_importances_:
            if i != 0 :
                a=a+1
        print(a)
    for key in test_dic.keys():
        res_dic[key]['score'] = model_tree[key].predict_proba(test_dic[key])[:, 1]
        res_dic[key]['score'] = res_dic[key]['score'].apply(lambda x: float('%.6f' % x))
        submission = pd.concat([submission,res_dic[key]])
    submission.to_csv('E:/dataset/preliminary_contest_data/submission.csv', index=False)
    # os.system('zip baseline.zip E:/dataset/preliminary_contest_data/submission.csv')


def main():
    parser = argparse.ArgumentParser(description='algorithm for tencent lookalike')
    parser.add_argument('--filesdir',type=str,default='E:/dataset/preliminary_contest_data')
    args = parser.parse_args()

    train_dic, train_label_dic, test_dic, res_dic = data_preprocess(args)
    patchextract(train_dic,train_label_dic,test_dic,res_dic)


if __name__=='__main__':
    main()