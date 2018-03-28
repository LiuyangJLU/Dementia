# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:35:55 2017
采用
@author: ly
"""
import pandas as pd
import numpy as np
import xgboost as xgb 
import math
import matplotlib.pylab as plt

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def prepare_data(filepath):
    sourcefile = pd.read_csv(filepath)
    temp = sourcefile.copy()
    DLB_count = 0
    nonDLB_count = 0
    for i in range(len(temp)):
        if temp.loc[i,'Diagnosis'] =='DLB':
            temp.loc[i,'Diagnosis']=1
            DLB_count+=1
        else:
            temp.loc[i,'Diagnosis']=0
            nonDLB_count+=1
    return temp,DLB_count,nonDLB_count


######### 分类器性能评价--交叉检验，混淆矩阵 #########
#统计TP,FP,TN,FN
def statistic(true,predict):
    TP=0 #TP：正确的正例
    TN=0 #TN：正确的负例
    FP=0 #FP：错误的正例
    FN=0 #FN：错误的负例
    for i in range(len(true)):
        if(true[i]==1):
            if(predict[i]==1):
                TP+=1 #真实为1，预测也为1
            else :
                FP+=1 #真实为1，预测为0
        elif(predict[i]==1):
            FN+=1 #真实为0，预测为1
        else :
            TN+=1 #真实为0，预测为0
            
    return [TP,FP,TN,FN]


#统计准确率衡量的5个指标：Sn,Sp,Avc,Acc,Mcc
def assess(TP,FP,TN,FN):
    Sn=Sp=Acc=Avc=Mcc=0 #评价分类器所用指标
    
    if(TP+FN!=0):
        Sn=TP*1.0/(TP+FN) #预测为1（正）的正确率
        
    if(TN+FP!=0):
        Sp=TN*1.0/(TN+FP) #预测为0（负）的正确率
        
    Avc=(Sn+Sp)*1.0/2 #正负平均准确率
    Acc=(TP+TN)*1.0/(TP+FP+TN+FN) #总体预测准确率
    
    if((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)!=0):
        Mcc=(TP*TN-FP*FN)*1.0/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)) 
         
    return [Sn,Sp,Acc,Avc,Mcc]


def kFoldTest(clf,raw_data,raw_label):
    predict=[]
    skf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in skf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        clf.fit(X_train, Y_train) 
        test_target_temp=clf.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target


        

def single_feature_acc(temp):
    '''单个特征的准确性，返回每个特征和所对应的acc'''
    label_array=temp.columns[:-1]#所有的特征列名
    label=list(temp.loc[:,'Diagnosis'])#标签列
    single_acc = {}
    for feature in label_array:
        dataset = temp.loc[:,feature].as_matrix(columns=None).reshape(-1,1)
        clf = SVC(kernel='rbf')
        values = statistic(label,kFoldTest(clf,dataset,label))
        Sn,Sp,Acc,Avc,Mcc = assess(values[0],values[1],values[2],values[3])
        
        single_acc[feature] = Acc
    return single_acc#返回单个特征的准确性

def select_add_one(temp):
    
    '''将各个特征进行分类的正确率排序，在进行一个特征一个特征的添加，删除正确率低的特征'''
    raw_labels = list(temp.loc[:,'Diagnosis'])
    single_acc = single_feature_acc(temp)
    #这里的fea_ascend是一个list类型，里面的值是元组类型，分别是特征名和准确性
    fea_ascend = sorted(single_acc.items(),key = lambda x :x[1],reverse = True )#内建的sorted返回的是一个新的List,并按照降序排列
    new_label_array = [fea[0] for fea in fea_ascend]
    base_Acc = fea_ascend[0][1]#访问第0行，第1列的元素值
    base_fea = [fea_ascend[0][0]]#访问第一个特征名
    for i in range(1,len(new_label_array)):
        base_fea.append(new_label_array[i])
        dataset = temp.loc[:,base_fea].as_matrix(columns=None)#数据集
        clf = SVC(kernel='rbf',C=1)
        values = statistic(raw_labels,kFoldTest(clf,dataset,raw_labels))
        Sn,Sp,Acc,Avc,Mcc = assess(values[0],values[1],values[2],values[3]) 
        
        if Acc < base_Acc:
            base_fea.remove(new_label_array[i])
        else:
            base_Acc = Acc
    base_fea.append('Diagnosis')
    return temp.loc[:,base_fea]

def xgboostfun(X,y):
    '''利用xgboost进行分类，X为数据集，y为标签'''
    a=[]
    skf = KFold(n_splits=10)
    confusion_matrix = [[0,0],[0,0]]    
    for  i,train_index,test_index  in skf.split(X,y):
        X_train,X_test = X[[train_index]],X[[test_index]]
        y_train,y_test = [y[i] for i in train_index],[y[i] for i in test_index]
        dtrain = xgb.DMatrix(X_train,label=y_train)
        dtest = xgb.DMatrix(X_test)
        params = {'max_depth':8,
                  'silent':0,
                  'min_child_weight':1,
                    'gamma ':0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这种样子。
                   'max_delta_step':1,#最大增量步长，我们允许每个树的权重估计。
                   'colsample_bytree':0.8, # 生成树时进行的列采样 
                    'objective':'binary:logistic',#定义需要被最小化的损失函数,binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
                   ' reg_lambda':1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                    'scale_pos_weight':1,
                    'n_estimators':200,#树的个数
                    'seed':1000}
        watchlist = [(dtrain,'train')]
        bst = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)#训练数据
        ypred = bst.predict(dtest)#预测数据
        
        # 设置阈值, 输出一些评价指标
        y_pred = (ypred >= 0.486)*1
        
        print('Auc: %.4f' %metrics.roc_auc_score(y_test,y_pred))
        print('Acc: %.4f' %metrics.accuracy_score(y_test,y_pred))
        a+=metrics.accuracy_score(y_test,y_pred)
        
        print('Recall: %.4f'%metrics.recall_score(y_test,y_pred))
        print('F1: %.4f' %metrics.f1_score(y_test,y_pred))
        print('Precision : %.4f' %metrics.precision_score(y_test,y_pred)) 
       

        what = metrics.confusion_matrix(y_test,y_pred)
        for k in range(2):
            for p in range(2):
                confusion_matrix[k][p] += what[k][p]
                
    return a/10


     
if __name__=='__main__':
    
    filepath =  r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    #filepath = r"~/Dementia/Apr_17th_DLB_nonDLB/Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    temp,DLB_count,nonDLB_count = prepare_data(filepath)
    temp_one = select_add_one(temp)
    raw_data = temp_one.drop('Diagnosis',1).as_matrix(columns=None)                 
    raw_target = list(temp_one.loc[:,'Diagnosis']) 
    print(xgboostfun(raw_data,raw_target))
        
        
        
        
        
        
        
        
        