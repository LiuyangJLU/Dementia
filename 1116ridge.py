# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:00:57 2017

@author: ly

"""
import numpy as np
import pandas as pd
import os
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt
import xgboost as xgb
import math

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from minepy import MINE


filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
temp=pd.DataFrame.from_csv(filepath, index_col=None)


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
raw_labels=list(temp.loc[:,'Diagnosis'])

#交叉验证均方误差函数， 评估获得最优参数  
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, raw_data, raw_labels, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)


alphas = [0.05, 0.1, 0.3,1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]




#cross_val_score函数的返回值就是对于每次不同的的划分raw data时，
#在test data上得到的分类的准确率。至于准确率的算法可以通过score_func参数指定，如果不指定的话，是用clf默认自带的准确率算法。



cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()#ridge误差率0.32160533823011811

#调用ridge进行
model_ridge = RidgeCV(alphas=[0.05, 0.1, 0.3,1, 3, 5, 10, 15, 30, 50, 75]).fit(raw_data,raw_labels)
print(model_ridge.alpha_)


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(raw_data,raw_labels)
print(model_lasso.alpha_)

#输出所选择的最优正则化参数情况下的残差平均值，因为是3折，所以看平均值
rmse_cv(model_lasso).mean()



coef_lasso = list(model_lasso.coef_)
feature_list=list(raw_data.columns)



coef = pd.Series(model_lasso.coef_, index = raw_data.columns)


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")




feature_coef = pd.DataFrame({'feature':feature_list,'coef':coef_lasso})
feature_coef = feature_coef.sort_index(by='coef',ascending=False)
#将特征权重结果写入Excel
writer = pd.ExcelWriter(r'E:\workspace\1118test\coef_descend.xlsx')
feature_coef.to_excel(writer,index=False)        
writer.save()




plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh",)
plt.title("Coefficients in the ridge Model")


imp_coef = pd.concat([feature_coef.sort_values(by='coef',ascending=False).head(10),feature_coef.sort_values
                   (by='coef',ascending=False).tail(10)])#取特征




#imp_coef = pd.concat([feature_coef.sort_values(by='coef',ascending=False).head(20),feature_coef.sort_values
#                      (by='coef',ascending=False).tail(15)])

#原始数据和原始标签
coef_Index=imp_coef['feature']
dataset = raw_data.loc[:,coef_Index].as_matrix(columns=None) #返回的不是numpy矩阵，而是numpy-array     
label = list(temp.loc[:,'Diagnosis'])




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
            
    writer = pd.ExcelWriter(r"E:\workspace\1120test\ACC3.xlsx") 
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

test_acc(dataset,label)       