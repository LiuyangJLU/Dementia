# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:43:03 2017

@author: ly
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt
import xgboost as xgb
import math

from sklearn import metrics
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix 


def prepare_data(filepath):
    
    filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    dataset=pd.read_csv(filepath, index_col=None)
    temp = dataset.copy()
    DLB_count = 0
    nonDLB_count = 0
    for i in range(len(temp)):
        if temp.loc[i,'Diagnosis'] == 'DLB':
            temp.loc[i,'Diagnosis'] = 1
            DLB_count+=1
        else:   
            temp.loc[i,'Diagnosis'] = 0
            nonDLB_count+=1
    return temp,DLB_count,nonDLB_count

#filepath =r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
#temp,DLB_count,nonDLB_count = prepare_data(filepath)
#raw_target = list(temp.loc[:,'Diagnosis'] ) 
#Label_Array = temp.columns[:-1]    
#import itertools
#combination = list(itertools.combinations(Label_Array,2))

def statistic(true,predict):
    TP=0 #TP：正确的正例
    TN=0 #TN：正确的负例
    FP=0 #FP：错误的正例
    FN=0 #FN：错误的负例
    for i in range(len(true)):
        if(true[i]==1):
            if(predict[i]==1):
                TP+=1 #真实为1，预测也为1
            else :
                FP+=1 #真实为1，预测为0
                
        elif(predict[i]==1):
            FN+=1 #真实为0，预测为1
        else :
            TN+=1 #真实为0，预测为0
            
    return [TP,FP,TN,FN]


#统计准确率衡量的5个指标：Sn,Sp,Avc,Acc,Mcc
def assess(TP,FP,TN,FN):
    Sn=Sp=Acc=Avc=Mcc=0 #评价分类器所用指标
    
    if(TP+FN!=0):
        Sn=TP*1.0/(TP+FN) #预测为1（正）的正确率
        
    if(TN+FP!=0):
        Sp=TN*1.0/(TN+FP) #预测为0（负）的正确率
        
    Avc=(Sn+Sp)*1.0/2 #正负平均准确率
    Acc=(TP+TN)*1.0/(TP+FP+TN+FN) #总体预测准确率
    
    if((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)!=0):
        Mcc=(TP*TN-FP*FN)*1.0/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)) 
         
    return [Sn,Sp,Acc,Avc,Mcc]

def kFoldTest(clf, raw_data, raw_target):
    '''
    十折交叉检验，clf是分类器，返回预测集
    '''
    predict=[]
    kf = KFold(n_splits=10)    
    for train_index, test_index in kf.split(raw_data):
        #print("TRAIN:", train_index, "TEST:", test_index)#查看如何分割数据
        X_train, X_test = raw_data[[train_index]], raw_data[[test_index]]
        #Y_test在这里没作用，为了数据变量对齐0.0
        Y_train, Y_test = raw_target[:test_index[0]]+raw_target[test_index[-1]+1:], raw_target[test_index[0]:test_index[-1]+1]        
        clf.fit(X_train,Y_train)
        test_target_temp=clf.predict(X_test)
        predict.append(test_target_temp)    
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target





def common_classier(raw_data, raw_target):
    '''
    使用常见的分类器进行分类
    '''
    from sklearn import neighbors  
    from sklearn.svm import SVC 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    df=pd.DataFrame(index=['Sn','Sp','ACC','AVC','MCC'])#该dataframe为了将正确率写入Excel

    clf = SVC(kernel='linear', C=1)
    value = statistic(raw_target,kFoldTest(clf, raw_data, raw_target))
    df['SVM']=assess(value[0],value[1],value[2],value[3])
    

    #用KNN
    clf=neighbors.KNeighborsClassifier(n_neighbors = 3 )
    value=statistic(raw_target,kFoldTest(clf, raw_data, raw_target))
    df['KNN']=assess(value[0],value[1],value[2],value[3])
    
#    #NB,pca的时候不能用
#    clf=MultinomialNB(alpha=0.01)
#    ACC,DLB,nonDLB=statistics(raw_target,kFoldTest(clf, ra
    
                                                    
    #Dtree
    clf = DecisionTreeClassifier(random_state=0)
    value=statistic(raw_target,kFoldTest(clf, raw_data, raw_target))
    df['Dtree']=assess(value[0],value[1],value[2],value[3])
    
    #随机森林
    clf = RandomForestClassifier(n_estimators= 30, max_depth=13, min_samples_split=110,  
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10)
    value=statistic(raw_target,kFoldTest(clf, raw_data, raw_target))
    df['RF']=assess(value[0],value[1],value[2],value[3])
    
    #boosting
    clf = AdaBoostClassifier(n_estimators=100)
    value=statistic(raw_target,kFoldTest(clf, raw_data, raw_target))
    df['adaboost']=assess(value[0],value[1],value[2],value[3])
    
    return df


'''
filepath = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
temp,DLB_count,nonDLB_count = prepare_data(filepath) 
raw_data = temp.drop('Diagnosis',1).as_matrix(columns=None)  
raw_label = list(temp.loc[:,'Diagnosis'])     
df = common_classier(raw_data,raw_label)

temp_acc = max(list(df.loc['ACC',:] ))
Label_Array = temp.columns[:-1]
import itertools
combination = list(itertools.combinations(Label_Array,2))
writer = pd.ExcelWriter(r'E:\workspace\Dementia\acc.xlsx')
i = 0
max_acc = 0 

for label_index in combination:
    sheet = "combination" + str(i)
    i += 1
    raw_data = temp.loc[:,label_index].as_matrix(columns=None)
    temp_df = common_classier(raw_data, raw_label)
    temp_acc = max(list(temp_df.loc['ACC',:] ))
    if temp_acc >= max_acc :
        temp_df.to_excel(writer,sheet_name=sheet,index=True)
        max_acc = temp_acc
writer.save()
'''
def Combination(temp, DLB_count, nonDLB_count, num):
    '''
    对数据集temp特征进行组合再进行分类
    '''
    raw_target = list(temp.loc[:,'Diagnosis'] ) 
    Label_Array = temp.columns[:-1]    
    import itertools
    combination = list(itertools.combinations(Label_Array,num))
    writer = pd.ExcelWriter(r'E:\workspace\Dementia\acc.xlsx')
    i = 0
    max_acc = 0
    for label_index in combination:
        sheet = "combination" + str(i)
        i += 1
        raw_data = temp.loc[:,label_index].as_matrix(columns=None)
        temp_df = common_classier(raw_data, raw_target)
        temp_acc = max(list(temp_df.loc['ACC',:] ))
        if temp_acc >= max_acc :
            temp_df.to_excel(writer,sheet_name=sheet,index=True)
            max_acc = temp_acc
    writer.save()
 




if __name__ == '__main__':
    
    filepath = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
    temp,DLB_count,nonDLB_count = prepare_data(filepath) 
    raw_data =  temp.drop('Diagnosis',1).as_matrix(columns=None)       
    raw_label = list(temp.loc[:,'Diagnosis']) 
    Combination(raw_data, DLB_count, nonDLB_count, 2)


    