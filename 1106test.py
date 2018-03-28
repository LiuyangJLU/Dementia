# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:24:59 2017
基于SVM-rfe
@author: ly
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.feature_selection import RFE
from sklearn import grid_search
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from minepy import MINE
filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
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
            temp.loc[i,'Diagnosis'] = 2
            VD_count+=1
        
raw_data=temp.drop('Diagnosis',1)
raw_label=list(temp.loc[:,'Diagnosis']) 


##原始数据，除掉类标的列

#类标所在的列     



def kFoldTest(estimator,raw_data,raw_label):
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)#对训练集进行训练
        test_target_temp=estimator.predict(X_test)#分类器进行预测
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



def svm_rfe_main(featurenum,raw_data,raw_label):
        
#    writer = pd.ExcelWriter(r"E:\workspace\result\svm-rfe_acc.xlsx")
#    df = pd.DataFrame(index=['ACC','VD_ACC','AD_ACC','DLB_ACC'])
    accuacy=[]
    Dtree_acc=[]
    Svm_acc=[]
    
    for num in featurenum:
        
        svc = SVC(kernel="linear", C=1)
        
        dataset=RFE(estimator=svc,n_features_to_select=num).fit_transform(raw_data,raw_label)
        
        
        print('决策树开始')
        tuned_parameters = [{'max_depth':[3,11,2],'criterion':['entropy','gini']}]
        Dtree = DecisionTreeClassifier()
        estimator = grid_search.GridSearchCV(Dtree,tuned_parameters,n_jobs=-1)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label))
        accuacy=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
        Dtree_acc.append(accuacy)
        
        df_acc1=pd.DataFrame(Dtree_acc,index=['ACC','DLB_ACC','VD_ACC','AD_ACC'],\
                             columns=['Dtree','SVM','KNN','RF','AdaBoost','NB'])
        
        tuned_parameters = [{'kernel':['rbf','linear'],'C':[]}]
        svm=SVC()
        estimator = grid_search.GridSearchCV(svm,tuned_parameters,n_jobs=-1)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label))
        df['svm']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
        
        
        
        #用KNN
        tuned_parameters = [{'n_neighbors':[3,50,2],'weights':['uniform','distance']}]#k值每次选择奇数个
        KNN=neighbors.KNeighborsClassifier()
        estimator = grid_search.GridSearchCV(KNN,tuned_parameters,n_jobs=-1)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label)) 
        df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        ##随机森林
        
        
        
        tuned_parameters = [{'n_estimators':[100,1000,100],'criterion':['entropy','gini']}]
        RF=RandomForestClassifier()
        estimator = grid_search.GridSearchCV(RF,tuned_parameters,n_jobs=-1)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label))
        df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        #AdaBoostClassifier
        estimator=AdaBoostClassifier(n_estimators=1000)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label))
        df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        
        #朴素贝叶斯
        estimator=BernoulliNB(alpha=0.01)
        ACC,DLB,VD,AD=Access(raw_label,kFoldTest(estimator,dataset,raw_label))
        df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        df.to_excel(writer,sheet_names=str(num)+'个特征数',index=True)
        writer.save()
        
svm_rfe_main([10,11,12,13,14,15],raw_data,raw_label)        
        
        
        






















