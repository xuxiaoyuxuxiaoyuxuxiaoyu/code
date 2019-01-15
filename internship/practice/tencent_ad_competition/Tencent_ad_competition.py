import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import argparse
import numpy as np

files=['adFeature.csv','userFeature.csv','train.csv','test1.csv']

def data_preprocess(args):
    adfeature_dir = os.path.join(args.filesdir,files[0])
    adfeature = pd.read_csv(adfeature_dir)
    aid_index = adfeature['aid']
    print('{} OK!'.format(files[0]))
    userfeature_dir = os.path.join(args.filesdir,files[1])
    if os.path.exists(userfeature_dir):
        userfeature = pd.read_csv(userfeature_dir)
    else:
        userfeature_data = []
        with open(os.path.join(args.filesdir,'userFeature.data'), 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userfeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            userfeature = pd.DataFrame(userfeature_data)
            userfeature.to_csv(os.path.join(args.filesdir,files[1]), index=False)
            del userfeature_data
            gc.collect()
            print('{} OK!'.format(files[1]))
    train = pd.read_csv(os.path.join(args.filesdir,files[2]))
    print('{} OK!'.format(files[2]))
    predict = pd.read_csv(os.path.join(args.filesdir, files[3]))
    print('{} OK!'.format(files[3]))

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, adfeature, on='aid', how='left')
    data = pd.merge(data, userfeature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))  # 对训练集的one-hot特征进行编码，0，1，2，3.。。
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]  # 提取训练集
    train_y = train.pop('label')  # 获取标签
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]  # 提取测试集
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
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
    test_x=pd.concat([res,test_x],axis=1)
    return train_x,train_y,aid_index,test_x

def patchextract(train_x,train_y,aid_index,test):
    train = pd.concat([train_x,train_y],axis=1)
    model_tree={}
    submission = np.array([])
    for aid in aid_index:
        train_single_data = train.loc[train['aid']==aid]
        train_single_label = train_single_data.pop('label')
        model_tree[aid] = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50, random_state=2018, n_jobs=100
        )
        model_tree[aid].fit(
        train_single_data, train_single_label,
        eval_set=[(train_single_data, train_single_label)], eval_metric='auc', early_stopping_rounds=100
        )
    for item in test:
        index = item.pop(['aid', 'uid'])
        index['score'] = model_tree[index['aid']].predict_proba(item)[:, 1]
        index['score'] = index['score'].apply(lambda x: float('%.6f' % x))
        submission = pd.concat([submission,index])
    submission.to_csv('./data/submission.csv', index=False)
    os.system('zip baseline.zip ./data/submission.csv')


def main():
    parser = argparse.ArgumentParser(description='algorithm for tencent lookalike')
    parser.add_argument('--filesdir',type=str,default='E:/dataset/preliminary_contest_data')
    args = parser.parse_args()

    train_x, train_y, aid_index, test_x = data_preprocess(args)
    patchextract(train_x, train_y, aid_index, test_x)


if __name__=='__main__':
    main()