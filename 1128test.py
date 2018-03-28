# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:06:24 2017
采用lassoCV进行特征选择，使用交叉验证均方误差函数，获得最佳参数，
使用xgaboost分类器
交叉验证采用留一法
@author: ly
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb


from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import plot_importance
#加载数据，统计标签数量
def prepare_data(filepath):
#    filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    dataset=pd.read_csv(filepath, index_col=None)
    temp = dataset.copy()
    DLB_count = 0
    nonDLB_count = 0
    for i in range(len(temp)):
        if temp.loc[i,'Diagnosis'] == 'DLB':
            temp.loc[i,'Diagnosis'] = 1
            DLB_count+=1
        else:   
            temp.loc[i,'Diagnosis'] = 0
            nonDLB_count+=1
    return temp,DLB_count,nonDLB_count

#filefullpath = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
#temp, DLB_count, nonDLB_count = prepare_data(filefullpath) 
#交叉验证均方误差函数， 评估获得最优参数  
#def rmse_cv(dataset,labels):
#    rmse= np.sqrt(-cross_val_score(model, dataset, labels, scoring="neg_mean_squared_error", cv = 3))
#    return(rmse)

def lassocv(dataset,labels):
    alpha = [1, 0.1, 0.001, 0.0005]
    model_lasso = LassoCV(alphas=alpha).fit(dataset,labels)
    coef_lasso = model_lasso.coef_
    return coef_lasso


def kfold(dataset,labels):
    skf = KFold(n_splits = 10)
    a = 0
    train_y_set = []
    test_y_set = []
    predict_set = []
    for train_index,test_index in skf.split(dataset,labels):       
        X_train,X_test = dataset[[train_index]],dataset[[test_index]]
        y_train,y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
        clf = XGBClassifier(
                                    silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
                                    learning_rate= 0.2, # 如同学习率
                                    min_child_weight=1,   #这个值过高会导致欠拟合
                                    max_depth=5, # 构建树的深度，越大越容易过拟合
                                    gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
                                    subsample=0.8, # 随机采样训练样本 训练实例的子采样比
                                    max_delta_step=1,#最大增量步长，我们允许每个树的权重估计。
                                    colsample_bytree=0.8, # 生成树时进行的列采样 
                                    nthread=4,
                                    objective= 'binary:logistic',
                                    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                    scale_pos_weight=1,
                                    n_estimators=200, #树的个数
                                    seed=1000 #随机种子
                                    )
        clf.fit(X_train,y_train,eval_metric='rmse')#eval_metric评价模型在测试集上的表现，
        y_true, y_pred = y_test, clf.predict(X_test)
        print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
        
        a += metrics.accuracy_score(y_true, y_pred)#因为是做十折交叉验证，返回的就是10次的ACC
        train_y_set.append(y_train)
        test_y_set.append(y_test)
        predict_set.append(y_pred)
    print(a/10)


    for temp in train_y_set :
        count0=0
        count1=0
        for i in temp:
            if i == 0:
                count0 += 1
            else:
                count1 += 1
        print(count0,count1)
    
if __name__ == '__main__':
    
    filefullpath = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    temp, DLB_count, nonDLB_count = prepare_data(filefullpath)  
    
    
    raw_data = temp.drop('Diagnosis',1)                
    raw_labels = list(temp.loc[:,'Diagnosis']) 
    lasso_coef=lassocv(raw_data,raw_labels)
#    error=rmse_cv(lasso_coef)
#    print('均方误差率:',error)
    
    feature_list=list(raw_data.columns)
    feature_coef = pd.DataFrame({'feature':feature_list,'coef':lasso_coef})
    feature_coef = feature_coef.sort_index(by='coef',ascending=False)
#    writer = pd.ExcelWriter(r'E:\workspace\data_result\coef_descend.xlsx')
#    feature_coef.to_excel(writer,index=False)        
#    writer.save()

    imp_coef = pd.concat([feature_coef.sort_values(by='coef',ascending=False).head(15),feature_coef.sort_values
                   (by='coef',ascending=False).tail(5)])#取特征
#    plt.rcParams['figure.figsize'] = (8.0, 10.0)
#    imp_coef.plot(kind = "barh",)
#    plt.title("Coefficients in the lasso Model")
    coef_Index=imp_coef['feature']
    
    dataset = raw_data.loc[:,coef_Index].as_matrix(columns=None) #返回的不是numpy矩阵，而是numpy-array     
    labels = list(temp.loc[:,'Diagnosis'])   
    kfold(dataset, labels)        
    
    
    
    
    
    
    