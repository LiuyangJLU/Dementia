# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:24:59 2017

@author: ly
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.feature_selection import RFE
from sklearn import grid_search
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


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

 #特征和类标           
raw_data=temp.drop('Diagnosis',1)
raw_label=list(temp.loc[:,'Diagnosis']) 




#利用主成分分析方法，选择20个特征数目


dataset=PCA(n_components=20).fit_transform(raw_data)


#利用10折交叉验证
#把数据集分为10份，一折做测试集，剩下做训练，分别做10次
def kFoldTest(estimator,raw_data,raw_label):      
        predict=[]
        kf=KFold(n_splits=10)
        #for循环主要是把数据索引
        for train_index,test_index in kf.split(raw_data):
            X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集合测试集
            #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
            #Y_test这个标签没用，只是做对称。
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




#feature_list = dataset.iloc[:,:53 ].columns.tolist()#所有的特征名



def main(raw_data,raw_label):
    writer=pd.ExcelWriter(r"E:\workspace\result\ACC_PCA.xlsx")
    df=pd.DataFrame(index=['ACC','DLB_ACC','VD_ACC','AD_ACC'])
        
    estimator = SVC(kernel='linear', C=1)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['SVM'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]


    #用KNN
    estimator = neighbors.KNeighborsClassifier(n_neighbors=3)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['KNN'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    ##随机森林
    estimator = RandomForestClassifier(n_estimators=10)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['RF'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值

    #AdaBoostClassifier
    estimator = AdaBoostClassifier(n_estimators=100)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['AdaBoost'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值

    estimator = BernoulliNB(alpha=1.0)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['BerNB'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    
    #决策树
    estimator = DecisionTreeClassifier(random_state=0)
    ACC,DLB,VD,AD = statistics(raw_label,kFoldTest(estimator,dataset,raw_label))
    df['DTree'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
    
    
main(raw_data,raw_label)
    
def picture()
#利用lasso选择重要性程度大的特征
def rank_importance_value(dataset,labels):
    
    selector = Lasso(alpha = 0.01)#使用lasso函数
    selector.fit(dataset,labels)
    dataset =dataset.iloc[:,abs(selector.coef_)!=0]#创建一个非零相关系数的列向量，
    
    return dataset


def get_name(name_index_dic, feature_list):

    result = []
    for num in feature_list:
        result.append(name_index_dic[num])

    return result 


    #删除最不重要的 N 个特征
def delete_feature(coefs,feature_name,k = 2):

    index_coefs = [(a,abs(coef)) for a,coef in zip(feature_name,coefs)]
    sorted_index_coefs = sorted(index_coefs,key = itemgetter(1),reverse = True)#获取对象第一个域的数据，

    for item in sorted_index_coefs[-k:]:
        feature_name.remove(item[0])  

    return feature_name


def recursive_elimination():











