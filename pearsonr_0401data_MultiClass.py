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
利用皮尔逊相关系数进行特征选择，
@author: ly
"""
import numpy as np
import pandas as pd

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
from scipy.stats import pearsonr
from math import sqrt
filepath=r'E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv'
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

raw_data=temp.drop('Diagnosis',1)
raw_labels=list(temp.loc[:,'Diagnosis'])
p_value=SelectKBest(lambda X, Y: np(map(lambda x:pearsonr(x, Y), X.T)).T, k=10).fit_transform(raw_data, raw_labels)















































