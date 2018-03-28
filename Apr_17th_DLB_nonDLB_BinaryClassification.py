# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:47:09 2017
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
    Dementia Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1 BinaryClassification
    T-test进行特征选择
    输出p_value
    cross_validation
    二分类分为0和1，把这个数据分为两份，
    一份是标签为0的，一份是标签为1的，
    T-Test在这里的作用就是可以计算每个特征把这两个标签分开的能力（对应P值越小）
    ，然后根据p值排序（升序）来选择特征
  优+优>优+劣   不一定成立
  读取p_value值排名前10特征的分类准确性
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


filename = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
dataset=pd.read_csv(filename)
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
        

#进行pValue升序排序
Label_Array=temp.columns[:-1]
pValue_Array=[]
for i in Label_Array:
    nonDLB = temp[temp['Diagnosis'] == 0][i]
    DLB = temp[temp['Diagnosis'] == 1][i]
    Tvalue,Pvalue=stats.ttest_ind(nonDLB,DLB)
    pValue_Array.append(Pvalue)
    
p=pd.DataFrame({'feaname':Label_Array,'pValue':pValue_Array})#利用字典的存储方式写到DataFrame里
p=p.sort_index(by='pValue',ascending=False)
#将排序结果写入Excel
writer = pd.ExcelWriter(r'E:\workspace\Dementia\p_value_DLB_nonDLB\P_value_descend.xlsx')
p.to_excel(writer,index=False)        
writer.save()         

        

#原始数据和原始标签
P_Index=p.head(10)['feaname']
raw_data = temp.loc[:,P_Index].as_matrix(columns=None) #返回的不是numpy矩阵，而是numpy-array                
raw_target = list(temp.loc[:,'Diagnosis'])



#k折交叉验证
def kFoldTest(estimator,raw_data,raw_target):      
    predict=[]
    kf=KFold(n_splits=10)
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称。
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
              
#writer={r"E:\workspace\Dementia\feature_select\p_value_before10_ACC.xlsx"}

#取P_value值排名前num的特征计算准确性
def P_Value_select_ACC(raw_data,raw_target):

    ACC_dict={}
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\p_value_DLB_nonDLB\p_value_before10_ACC.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','nonDLB_ACC'])
    
     #利用各个分类器去测试准确性
     #用SVM
    estimator = SVC(kernel='linear', C=1)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['SVM']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]

    #用KNN
    estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['KNN']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    ##随机森林
    estimator=RandomForestClassifier(n_estimators=10)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['RF']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['AdaBoost']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=MultinomialNB(alpha=0.01)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['NB']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值


     #神经网络
    estimator=MLPClassifier(alpha=1)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['MLP']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值

   #决策树
    estimator=DecisionTreeClassifier(random_state=0)
    ACC,nonDLB,DLB=statistics(raw_target,kFoldTest(estimator,raw_data,raw_target))
    df['DTree']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    ACC_dict=df
    df.to_excel(writer,'p_value排名前10的特征',index=True)
    writer.save()
    return  ACC_dict
test=P_Value_select_ACC(raw_data,raw_target)
    
    
    
    
    
    
    
    
    
    
    
    
    