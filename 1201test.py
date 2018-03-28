# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:56:36 2017

@author: ly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import math

from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


#加载数据，统计标签数量
def prepare_data(filepath):
    filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
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

def leaveoneout(dataset,labels):
    '''分类器采用xgboost,交叉验证采用留一法'''
    leaveoo =LeaveOneOut()
    Y_true = [] 
    Y_pred  = []   
    #xgboost参数分为三类：
    '''1、通用参数
       2、Booster参数:控制每一步的booster
       3、学习目标参数：控制训练目标的表现'''
    for train_index , test_index in leaveoo.split(dataset):
            x_train,x_test = dataset[[train_index]],dataset[[test_index]]
            y_train,y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
            clf = XGBClassifier(       
                                    max_depth=6,
                                    silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
                                    min_child_weight=1,  #孩子接节点最小的样本权重和                            
                                    gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这种样子。
                                    max_delta_step=1,#最大增量步长，我们允许每个树的权重估计。
                                    colsample_bytree=0.8, # 生成树时进行的列采样 
                                    nthread=4,
                                    objective= 'binary:logistic',#定义需要被最小化的损失函数,binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
                                    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                    scale_pos_weight=1,
                                    n_estimators=200, #树的个数
                                    seed=1000 #随机种子
                                )   
            clf.fit(x_train,y_train,eval_metric='auc')
            y_true, y_pred = y_test, list(clf.predict(x_test))
            Y_true.append(y_test[0])
            Y_pred.append(y_pred[0])
            print(y_true,y_pred)
    print("Accuracy : %.6g" % metrics.accuracy_score(Y_true ,Y_pred))

def exe(raw_data,raw_labels):
    
   filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
   temp,DLB_count,nonDLB_count = prepare_data(filepath)
   
   dataset = temp.drop('Diagnosis',1).as_matrix(columns=None)   
   labels = list(temp.loc[:,'Diagnosis'])
   
   leaveoneout(dataset,labels)
    
    
    
    
    
    