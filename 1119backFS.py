# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:00:57 2017

@author: ly

"""
import numpy as np
import pandas as pd
import os
import math
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt


from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	



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

filepath = r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
temp,DLB_count,nonDLB_count = prepare_data(filepath)
dataset = temp.drop('Diagnosis',1).as_matrix(columns=None)
labels = temp.iloc[:,temp.shape[1]-1].values.astype('int')




#评价函数(交叉验证)

def get_acc(dataset,label):
    scores = []
    skf=StratifiedKFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in skf.split(dataset,label):
        X_train,X_test = dataset[train_index],dataset[test_index]#将数据分为验证集和测试集
        Y_train, y_test = label[train_index],label[test_index]
        clf = SVC(kernel='rbf',C=10).fit(X_train,Y_train)
        scores.append(clf.score(X_test, y_test))
    return np.mean(scores)



                              
def backFS(dataset,labels,func,n_features=0):
    
    score_ori = func(dataset,labels)#起始得分
    index_kicked = []#排名
    scores = []
    fea_num_ori = dataset.shape[1] - n_features#原始特征数
    dataset = np.vstack((dataset,np.arange(dataset.shape[1])))
    for _ in range(fea_num_ori - 1):#这里的_符号表示只循环range里面那么多次，就不用管了。
        result =[-1,0,0]#第一个参数的存放原矩阵列号，第二个参数代表score,第三个参数代表子矩阵的列号
        for i in range(dataset.shape[1]):#循环所有特征
            subset=np.delete(dataset,i,axis=1)#每次删除一个特征
            score = func(subset[:-1,:],labels)#将删掉的特征进行训练
            
            if score >result[1]:#如果删掉特征后的score大于原始acc
                
                result[0] = dataset[-1,i]#从前往后存放列号
                result[1] = score
                result[2] = i
                
        index_kicked.append(result[0])  #原矩阵的列号 
        scores.append(result[1])
        dataset = np.delete(dataset,result[2],axis=1)
         
        if dataset.shape[1] == 1:#如果特征数只有1个，停止进行排序
            index_kicked.append(dataset[-1,0])
            scores.insert(0,score_ori)#将得分依次插入到scores列表里
        continue
#        index_selected = list(dataset[-1,:])
#        dataset = dataset[:-1,:]
        if n_features ==0:
            dataset = []
        list(reversed(index_kicked))#反向排序，将得分低的特征所在的列排在后面
        list(reversed(scores))#反向排序    #特征得分反序
        
    return  dataset,index_kicked,scores

dataset,ranking,scores=backFS(dataset,labels,get_acc)
ranking = list(map(int,ranking))


dataset=dataset[:,ranking[:15]]
labels=list(temp.loc[:,'Diagnosis'])




    
######### 分类器性能评价--交叉检验，混淆矩阵 #########

#统计TP,FP,TN,FN
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


def kFoldTest(estimator,dataset,label):
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(dataset):
        X_train,X_test = dataset[train_index],dataset[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = label[:test_index[0]]+label[test_index[-1]+1:], label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)
        test_target_temp=estimator.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target


def test_acc(dataset,label):
        
    writer = pd.ExcelWriter(r"E:\workspace\1120test\ACC4.xlsx") 
    df=pd.DataFrame(index=['Sn','Sp','ACC','AVC','MCC'])
     #利用各个分类器去测试准确性
     #用SVM
    estimator = SVC(kernel='linear', C=10)
    value = statistic(label,kFoldTest(estimator,dataset,label))
    df['SVM-linear'] = assess(value[0],value[1],value[2],value[3])
    
    
    estimator=SVC(kernel='rbf',C=10)#C值=10效果最好
    value = statistic(label,kFoldTest(estimator,dataset,label))
    df['SVM_rbf'] = assess(value[0],value[1],value[2],value[3])
            
    
    estimator=SVC(kernel='sigmoid',C=10)
    value = statistic(label,kFoldTest(estimator,dataset,label))
    df['SVM_sigmoid']= assess(value[0],value[1],value[2],value[3])
   
    #用KNN
    estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
    value=statistic(label,kFoldTest(estimator,dataset,label))
    df['KNN'] = assess(value[0],value[1],value[2],value[3])

    
    ##随机森林
    estimator=RandomForestClassifier(n_estimators=10)
    value = statistic(label,kFoldTest(estimator,dataset,label))
    df['RF'] = assess(value[0],value[1],value[2],value[3])
    
    
    #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    value=statistic(label,kFoldTest(estimator,dataset,label))
    df['Adaboost']=assess(value[0],value[1],value[2],value[3])
    
    #朴素贝叶斯
    estimator=MultinomialNB(alpha=0.01)
    value = statistic(label,kFoldTest(estimator,dataset,label))
    df['NB']=assess(value[0],value[1],value[2],value[3])
    
    
    #决策树
    estimator=DecisionTreeClassifier(random_state=0)
    value=statistic(label,kFoldTest(estimator,dataset,label))
    df['Dtree'] = assess(value[0],value[1],value[2],value[3])
    df.to_excel(writer,index=True)
    
    writer.save()

test_acc(dataset,labels)       


#==============================================================================
#     df_svm_linear = pd.DataFrame(scores_svm_linear,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_svm_rbf = pd.DataFrame(score_svm_rbf,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_svm_sigmoid = pd.DataFrame(score_svm_sigmoid,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_KNN = pd.DataFrame(scores_knn,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_RF = pd.DataFrame(scores_rf,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_Adaboost = pd.DataFrame(scores_adaboost,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_DTree = pd.DataFrame(scores_dtree,columns=['Sn','Sp','ACC','AVC','MCC'])
#     df_NB = pd.DataFrame(scores_nb,columns=['Sn','Sp','ACC','AVC','MCC'])   
#     
#     
#     df_svm_linear.to_excel(writer,'svm_linear',index=True)
#     df_svm_rbf.to_excel(writer,'svm_rbf',index=True)
#     df_svm_sigmoid.to_excel(writer,'svm_sigmoid',index=True)
#     df_KNN.to_excel(writer,'KNN',index=True)
#     df_RF.to_excel(writer,'RF',index=True)
#     df_Adaboost.to_excel(writer,'Adaboost',index=True)
#     df_DTree.to_excel(writer,'Dtree',index=True)
#     df_NB.to_excel(writer,'NB',index=True)
#==============================================================================




