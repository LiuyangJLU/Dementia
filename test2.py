# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:05:21 2017

@author: ly
"""

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import KFold

def prepare_data(filepath):
    filepath=r"E:\data2.csv"
    dataset=pd.read_csv(filepath, index_col=None)
    label1_count=0
    label0_count=0
    temp = dataset.copy()
    for i in range(len(temp)):
        if temp.loc[i,'label'] == 1:
            label1_count+=1
        else:   
            temp.loc[i,'label'] = 0
            label0_count+=1
    return temp,label1_count,label0_count


def select_PCA(temp, num):
    '''
    PCA主成分分析法
    '''
    #原始数据和原始标签
    raw_data = temp.drop('label',1).as_matrix(columns=None)                 
    raw_target = list(temp.loc[:,'label'] ) 
    from sklearn.decomposition import PCA

    #主成分分析法，返回降维后的数据
    #参数n_components为主成分数目
    fea_data = PCA(n_components = num).fit_transform(raw_data)
    return fea_data, raw_target


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
#    ACC,DLB,nonDLB=statistics(raw_target,kFoldTest(clf, raw_data, raw_target))
#    print(ACC)
    
    #Dtr
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



def common_classier_exe(temp, label1_count, label0_count):
    '''
    使用各种特征选择算法，再用常见的分类器进行分类
    '''
    writer = pd.ExcelWriter(r'E:\ACC.xlsx') 
    data, label = select_PCA(temp, 30)
    df = common_classier(data, label)
    df.to_excel(writer,sheet_name="PCA",index=True)
    writer.save()


if __name__=='__main__':
    
    filepath = r'E:\data2.csv'
    temp,label1_count,label0_count = prepare_data(filepath)
    raw_data = temp.drop('label',1).as_matrix(columns=None)                 
   
    
    temp_df = common_classier(raw_data, raw_label)      
    common_classier_exe(temp, label1_count, label0_count)
    
