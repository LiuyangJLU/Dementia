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
利用递归特征消除法，
@author: ly
"""
import numpy as np
import pandas as pd
import os


from scipy import stats#导入统计的包T_test检验
from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest	

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
#filepath="~/Dementia/Apr_1th_AD_VD_DLB/Q_AD_DLB_VD_after_normalizion_0_to_1.csv"

dataset=pd.read_csv(filepath)
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
raw_data = temp.drop('Diagnosis',1)
#类标所在的列     
raw_label=list(temp.loc[:,'Diagnosis']) 
#raw_newdata=RFE(estimator=LogisticRegression(), n_features_to_select=10).fit_transform(raw_data, raw_label)


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


def RFE_Acc(featurenum,raw_data,raw_label):
    #writer = pd.ExcelWriter(r"E:\workspace\Dementia\test_acc\ACC.xlsx")
    writer = pd.ExcelWriter(r"E:\workspace\result\ACC.xlsx") 

    df=pd.DataFrame(index=['ACC','DLB_ACC','AD_ACC','VD_ACC'])
    for num in featurenum:
        #递归特征消除
        #selector = RFE(estimator, 3000, step=200)#每次迭代删除200个特征，保留3000个特征
        ##使用逻辑斯特回归分类器，具有coef_属性,在训练一个评估器后,coef_相当于每个特征的权重，RFE在每一次迭代中得到一个新的coef_
        #并删除权重低的特征，直到剩下的特征达到指定的数量
        #n_features_to_select选择的特征个数
        raw_newdata=RFE(estimator=LogisticRegression(), n_features_to_select=num).fit_transform(raw_data, raw_label)
    
    #利用各个分类器去测试准确性
     #用SVM
        estimator = SVC(kernel='linear', C=1)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['SVM']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
        
        #用KNN
        estimator=neighbors.KNeighborsClassifier(n_neighbors=10)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        ##随机森林
        estimator=RandomForestClassifier(n_estimators=1000)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        #AdaBoostClassifier
        estimator=AdaBoostClassifier(n_estimators=1000)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        #朴素贝叶斯
        estimator=MultinomialNB(alpha=0.01)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        
         #神经网络
        estimator=MLPClassifier(alpha=1)
        ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['MLP']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    df.to_excel(writer,sheet_name=str(num)+'个特征',index=True)
    writer.save()
RFE_Acc([5,6,7,8,9,10],raw_data,raw_label)
    
    
    
    
    








