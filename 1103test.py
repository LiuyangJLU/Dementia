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
Q_AD_DLB_VD_after_normalizion_0_to_三分类问题


@author: ly
"""
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.linear_model import Lasso
from operator import  itemgetter


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
#filepath="~/Dementia/Apr_1th_AD_VD_DLB/Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
dataset=pd.DataFrame.from_csv(filepath, index_col=None)
temp=dataset.copy()


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
##原始数据，除掉类标的列
dataset=temp.drop('Diagnosis',1)
#类标所在的列     
raw_label=list(temp.loc[:,'Diagnosis'])

 
  


#利用lasso选择重要性程度大的特征
def rank_importance_value(dataset,labels):
    
    selector = Lasso(alpha = 0.01)#使用lasso函数
    selector.fit(dataset,labels)
    dataset =dataset.iloc[:,abs(selector.coef_)!=0]#创建一个非零相关系数的列向量，
    
    return dataset


raw_data = rank_importance_value(dataset,raw_label).as_matrix(columns=None)


def kFoldTest(estimator,raw_data,raw_label):
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)
        test_target_temp=estimator.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target
 

#函数功能：统计正确率，以及每个类的正确数
def statistics(raw,test):
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

#writer=pd.ExcelWriter(r"E:\workspace\Dementia\result\acc.xlsx")
#df=pd.DataFrame(index=['ACC','DLB_ACC','VD_ACC','AD_ACC'])

####使用网格搜索进行超参数调优#############################
#待调参数组成的网格，类似暴力搜索
print('KNN开始')
 #KNN
tuned_parameters = [{'n_neighbors':[3,50,2],'weights':['uniform','distance']}]#k值每次选择奇数个
KNN=neighbors.KNeighborsClassifier()
estimator = grid_search.GridSearchCV(KNN,tuned_parameters,n_jobs=-1)
ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
print("KNN的准确性:",ACC,DLB/DLB_count,VD/VD_count,AD/AD_count)#返回的每一列的值
print('KNN结束')


print('SVM开始')
#SVM
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
svr = SVC()#使用svm
estimator = grid_search.GridSearchCV(svr,tuned_parameters,n_jobs=-1)#n_hobs设置为-1，表示cpu里面的所有core进行工作
ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
print('SVM的准确性:',ACC,DLB/DLB_count,VD/VD_count,AD/AD_count)
print('SVM结束')


print('DTree开始')
 #DTree
tuned_parameters = [{'max_depth':[3,11,2],'criterion':np.array(['entropy','gini'])}]
Dtree = DecisionTreeClassifier()
estimator = grid_search.GridSearchCV(Dtree,tuned_parameters)
ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
print('DTtree的准确性:',ACC,DLB/DLB_count,VD/VD_count,AD/AD_count)
print('DTree结束')


print('RF开始')
##随机森林
tuned_parameters = [{'n_estimators':[100,1000,100],'criterion':np.array(['entropy','gini'])}]
RF=RandomForestClassifier()
estimator = grid_search.GridSearchCV(RF,tuned_parameters,n_jobs=-1)
ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
print('RF的准确性:',ACC,DLB/DLB_count,VD/VD_count,AD/AD_count)#返回的每一列的值
print('RF结束')

print('朴素贝叶斯开始')
 #朴素贝叶斯
estimator = BernoulliNB()
ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
print('朴素贝叶斯的准确性:',ACC,DLB/DLB_count,VD/VD_count,AD/AD_count)#返回的每一列的值
print('朴素贝叶斯结束')

