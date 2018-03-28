# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:08:05 2017

@author: ly
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from minepy import MINE


filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
temp=pd.DataFrame.from_csv(filepath, index_col=None)

#1代表DLB，0代表nonDLB,替换原标签并计数
DLB_count=0
nonDLB_count=0
for i in range(len(temp)):
    if temp.loc[i,'Diagnosis'] =='DLB' :
        temp.loc[i,'Diagnosis']=1
        DLB_count+=1#DLB_count=300
    else :
        temp.loc[i,'Diagnosis']=0
        nonDLB_count+=1#nonDLB=1049
#特征和类标           
raw_data=temp.drop('Diagnosis',1)
raw_label=list(temp.loc[:,'Diagnosis'])


def MIC(x,y):
    mine = MINE()
    mine.compute_score(x,y)
    micValue = mine.mic()
    
    return micValue

#MIC(a,raw_label)
#MIC(b,raw_label)
#raw_data.shape[1]
#c=raw_data.ix[:,2]
#micvalue=MIC(raw_data.ix[:,4],raw_label)
micvalue = MIC(raw_data.ix[:,4],raw_label)
def McOne(F,C,threshold=0.005):

    featureNum = F.shape[1]#特征数
    micFC = []#存放筛选出来的特征
    
    for i in range(featureNum):
        micValue = MIC(F.ix[:,i],C)
        if micValue >= threshold:
            micFC.append(micValue)
    micFC = np.array(micFC)#以数组形式存放
    subset = range(micFC.size)
    subset = np.argsort(-micFC[subset])#排序
    subset = list(subset)#存放列表
    numSubset = len(subset)#特征子集个数
    
    
    for e in range(numSubset):
        q= e+1
    while q < numSubset:
        if MIC(F.ix[:,subset[e]],F.ix[:,subset[q]]) >= micFC[subset[q]]:
            del subset[q]#移除与已知特征没有相关性的特征
            numSubset = numSubset-1
        else:
            q +=1
        e +=1        
    return F.ix[:,subset]


dataset=McOne(raw_data,raw_label).as_matrix(columns=None)#数据集
label = list(temp.loc[:,'Diagnosis'])#标签


def kFoldTest(estimator,raw_data,raw_label):      
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)#
        test_target_temp=estimator.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target


#函数功能：统计正确率，以及每个类的正确数
def Access(raw,test):
    nonDLB_true=0
    DLB_true=0
    for i in range(len(raw)):
        if raw[i]==0 and test[i]==0:
            nonDLB_true+=1
        if raw[i]==1 and test[i]==1:
            DLB_true+=1
    ACC=(DLB_true+nonDLB_true)/len(raw)
    return ACC,nonDLB_true,DLB_true 

def main(dataset,label):
    writer=pd.ExcelWriter(r"E:\workspace\1119test\Acc.xlsx")
    df=pd.DataFrame(index=['ACC','DLB_ACC','nonDLB_ACC'])
        
    estimator=SVC(kernel='rbf',C=10)#C值=10效果最好
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_rbf'] = [ACC,nonDLB/nonDLB_count,DLB/DLB_count]


    #用KNN
    estimator=SVC(kernel='poly',C=10)
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_poly'] =[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    ##随机森林
    estimator = RandomForestClassifier(n_estimators=10)
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_linear'] =[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值\
    
    estimator=SVC(kernel='sigmoid',C=10)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_sigmoid']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]
    
    #RF
    estimator=RandomForestClassifier(n_estimators=100,criterion="entropy")
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['RF']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]
    
     #用KNN
    
    estimator = neighbors.KNeighborsClassifier(n_neighbors=10)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['KNN']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
  #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['AdaBoost']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=BernoulliNB(alpha=0.01)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['NB']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
    
    
main(dataset,label)
