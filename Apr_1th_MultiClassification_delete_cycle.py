# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:52:43 2017
Python3
Required Packages
--pandas
--numpy
--sklearn
--scipy

Info
-name   :"liuyang"
-email  :'1410455465@qq.com'
-date   :2017-10-24
-Description 
利用循环删除的方法进行特征选择和组合
@author: ly
"""
import numpy as np
import pandas as pd
import os


from scipy.stats import ttest_ind_from_stats#导入统计的包T_test检验
from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

filename = r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
dataset  = pd.read_csv(filename)#读数据
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
raw_data=temp.drop('Diagnosis',1)
#类标所在的列
raw_target=list(temp.loc[:,'Diagnosis']) 


#利用10折交叉验证
#把数据集分为10份，一折做测试集，剩下做训练，分别做10次
def kFoldTest(estimator,raw_data,raw_target):      
        predict=[]
        kf=KFold(n_splits=10)
        #for循环主要是把数据索引
        for train_index,test_index in kf.split(raw_data):
            X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集合测试集
            #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
            #Y_test这个标签没用，只是做对称。
            Y_train, Y_test = raw_target[:test_index[0]]+raw_target[test_index[-1]+1:], raw_target[test_index[0]:test_index[-1]+1] 
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

label_array = list(raw_data.columns)
#循环删除的思路：
#将要删除特征数按照从前往后进行删除，每次删除的个数任意设置，
def delete_cycle(num,raw_data,raw_target):
    label_array=list(raw_data.columns)#获取所有列标签返回list
    i=0
    ACC_dict={}
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\cycle_delete\ACC_cycle_delete.xlsx")
    while(i<len(label_array)):
        #数据集的划分是通过索引的方式，从起始位置0开始一直到想要分割的数(num)为止，一次分10个特征
        #loc[]表示取训练集里面数据的多少行，里面的数据从0到i到特征列表里最小选择的特征数。
        train_data=raw_data.loc[:,label_array[0:i]+label_array[min(i+num,len(label_array)):]].as_matrix(columns=None)
        i+=num#每次累加10次
        df = pd.DataFrame(index=['ACC','DLB_ACC','AD_ACC','VD_ACC'])##创建一个DataFrame,为了方便后续存到表里
        #利用各个分类器去测试准确性
        #用SVM
        estimator = SVC(kernel='linear', C=1)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['SVM'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    
        #用KNN
        estimator = neighbors.KNeighborsClassifier(n_neighbors=3)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['KNN'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
        
        ##随机森林
        estimator = RandomForestClassifier(n_estimators=10)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['RF'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
        #AdaBoostClassifier
        estimator = AdaBoostClassifier(n_estimators=100)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['AdaBoost'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
        #朴素贝叶斯
        estimator = MultinomialNB(alpha=0.01)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['NB'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值


        #神经网络
        estimator = MLPClassifier(alpha=1)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['MLP'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
        #决策树
        estimator = DecisionTreeClassifier(random_state=0)
        ACC,DLB,VD,AD = statistics(raw_target,kFoldTest(estimator,train_data,raw_target))
        df['DTree'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
        ACC_dict[i-num] = df
        df.to_excel(writer,'删除'+str(i-num)+'--'+str(min(i,len(label_array)))+'特征',index=True)
    writer.save()
    return ACC_dict
test=delete_cycle(10,raw_data,raw_target)       
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        