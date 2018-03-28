# -*- coding: utf-8 -*-
"""
Python3
Required Packages
--pandas
--numpy
--sklearn
--scipy

Info
-name   :"liuyang"
-email  :'1410455465@qq.com'
-date   :2017-10-28
-Description 
二分类
利用卡方检验的方法进行特征选择，通过卡方值来判断特征是否与类型有关。
卡方值越小，说明越不相关，特征需要去除。
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
from sklearn.feature_selection import chi2



filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
dataset=pd.read_csv(filepath)
temp=dataset.copy()

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
raw_data=temp.drop('Diagnosis',1)
raw_target = list(temp.loc[:,'Diagnosis'])
raw_newdata = SelectKBest(chi2, k=15).fit_transform(raw_data, raw_target)#利用卡方检验，选择10个最好的特征并返回

    
def kFoldTest(estimator,raw_newdata,raw_target):      
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_newdata):
        X_train,X_test = raw_newdata[train_index],raw_newdata[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_target[:test_index[0]]+raw_target[test_index[-1]+1:], raw_target[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)#
        test_target_temp=estimator.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target

def statistics(raw,test):
    nonDLB_true=0
    DLB_true=0
    for i in range(len(raw)):
        if raw[i]==0 and test[i]==0:
            nonDLB_true+=1
        if raw[i]==1 and test[i]==1:
            DLB_true+=1
    ACC=(DLB_true+nonDLB_true)/len(raw)
    return ACC,nonDLB_true,DLB_true 
def chi2_test_ACC(raw_newdata,raw_target):

    ACC_dict={}
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\chi2_select\chi2_15feature_ACC.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','nonDLB_ACC'])
    
     #利用各个分类器去测试准确性
     #用SVM
    estimator = SVC(kernel='linear', C=1)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['SVM']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]

    #用KNN
    estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['KNN']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    ##随机森林
    estimator=RandomForestClassifier(n_estimators=10)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['RF']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['AdaBoost']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=MultinomialNB(alpha=0.01)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['NB']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值


     #神经网络
    estimator=MLPClassifier(alpha=1)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['MLP']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值

   #决策树
    estimator=DecisionTreeClassifier(random_state=0)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_newdata,raw_target))
    df['DTree']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    ACC_dict=df
    df.to_excel(writer,'卡方检验选择15个特征的ACC',index=True)
    writer.save()
    return  ACC_dict
test=chi2_test_ACC(raw_newdata,raw_target)




















       