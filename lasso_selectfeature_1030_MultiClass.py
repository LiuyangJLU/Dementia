# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 08:52:43 2017
Python3
Required Packages
--pandas
--numpy
--sklearn
--scipy

Info
-name   :"liuyang"
-email  :'1410455465@qq.com'
-date   :2017-10-29
-Description 
Embedded思想
基于惩罚项的特征选择


"""
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn import grid_search
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.linear_model import LogisticRegression


filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
temp=pd.read_csv(filepath)

 ##加载类标签，并对每个类别进行计数
VD_count=0
AD_count=0
DLB_count=0  
for i in range(len(temp)):
    if temp.loc[i,'Diagnosis'] =='DLB' :
            temp.loc[i,'Diagnosis']=0
            DLB_count+=1
    elif temp.loc[i,'Diagnosis'] =='AD' :            
            temp.loc[i,'Diagnosis']=1
            AD_count+=1
    else :
            temp.loc[i,'Diagnosis']=2
            VD_count+=1

#原始数据
raw_data=temp.drop('Diagnosis',1)
raw_label=list(temp.loc[:,'Diagnosis'])
    
#交叉验证       
def kfold(estimator,raw_data,raw_label):
    predict=[]
    kf=KFold(n_splits=10)
    for train_index,test_index in kf.split(raw_data):
         X_train,X_test=raw_data[train_index],raw_data[test_index]
         Y_train, Y_test=raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1]
         estimator.fit(X_train, Y_train)#
         test_target_temp=estimator.predict(X_test)#用测试集让模型来训练预测
         predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target



def Access(raw,test):
    DLB_true=0
    AD_true=0
    VD_true=0
    for i in range(len(raw)):
        if raw[i]==0 and test[i]==0:
            DLB_true+=1
        if raw[i]==1 and test[i]==1:
            AD_true+=1
        if raw[i]==2 and test[i]==2:
            VD_true+=1
    ACC=(DLB_true+AD_true+VD_true)/len(raw)
    return ACC,DLB_true,AD_true,VD_true

def select_feature(raw_data,raw_label):
    
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\penalty_select\ACC_L15.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','AD_ACC','VD_ACC'])
    #带L1惩罚项的逻辑斯特回归作为基模型的特征选择，保留多个对目标值具有同等相关性中的一个，
    raw_newdata=SelectFromModel(LogisticRegression(penalty='l1',C=1.0)).fit_transform(raw_data, raw_label)
    print(np.shape(raw_newdata))
    #kernel='rbf'
    estimator=SVC(kernel='rbf',C=10)#C值=10效果最好
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['SVM_rbf']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #kernel='poly'
    estimator=SVC(kernel='poly',C=10)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['SVM_poly']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    
    #kernel='linear'
    estimator=SVC(kernel='linear',C=10)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['SVM_linear']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #kernel='sigmoid'
    
    estimator=SVC(kernel='sigmoid',C=10)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['SVM_sigmoid']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #RF
    estimator=RandomForestClassifier(n_estimators=100,criterion="entropy")
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
     #用KNN
    
    estimator = neighbors.KNeighborsClassifier(n_neighbors=10)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label)) 
    df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
  #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=BernoulliNB(alpha=0.01)
    ACC,DLB,VD,AD=Access(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
    df.to_excel(writer,index=True)
    writer.save()
    
select_feature(raw_data,raw_label)
    

















