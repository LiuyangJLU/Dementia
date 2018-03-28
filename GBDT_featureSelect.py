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
-date   :2017-11-01
-Description 

基于树模型的特征选择
GBDT

"""
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier


filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
temp=pd.read_csv(filepath)


##加载类标签，并对每个类别进行计数
DLB_count=0
AD_count=0
VD_count=0
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
raw_data=temp.drop('Diagnosis',1).as_matrix(columns=None)   
raw_label=list(temp.loc[:,'Diagnosis'])


def kfold(estimator,raw_data,raw_lable):
    predict=[]
    kf=KFold(n_splits=10)
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test=raw_data[train_index],raw_data[test_index]
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train,Y_train)
        test_target=estimator.predict(X_test)
        predict.append(test_target)
        test_target=[i for temp in predict for i in temp]
    return test_target
    
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

def gbdt_select(raw_data,raw_label):
    
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\gbdt_select\ACC.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','AD_ACC','VD_ACC'])
    
    #GBDT作为基模型的特征选择
    raw_newdata=SelectFromModel(GradientBoostingClassifier()).fit_transform(raw_data, raw_label)
    

      #利用各个分类器去测试准确性
      #用SVM
    estimator = SVC(kernel='linear', C=1)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['SVM']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #用KNN
    estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    ##随机森林
    estimator=RandomForestClassifier(n_estimators=10)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=MultinomialNB(alpha=0.01)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    
     #神经网络
    estimator=MLPClassifier(alpha=1)
    ACC,DLB,VD,AD=statistics(raw_label,kfold(estimator,raw_newdata,raw_label))
    df['MLP']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
    

gbdt_select(raw_data,raw_label)





















