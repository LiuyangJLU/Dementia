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
Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1二分类问题
利用递归特征消除法，
@author: ly
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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


##原始数据，除掉类标的列
raw_data=temp.drop('Diagnosis',1)
#类标所在的列
raw_label=list(temp.loc[:,'Diagnosis']) 


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
    nonDLB_true=0
    DLB_true=0
    for i in range(len(raw)):
        if raw[i]==0 and test[i]==0:
            nonDLB_true+=1
        if raw[i]==1 and test[i]==1:
            DLB_true+=1
    ACC=(DLB_true+nonDLB_true)/len(raw)
    return ACC,nonDLB_true,DLB_true 

def RFE_Acc(featurenum,raw_data,raw_label):
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\RFE_select\二分类_select10-18_ACC.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','nonDLB_ACC'])
    for num in featurenum:
        #递归特征消除
        #selector = RFE(estimator, 3000, step=200)#每次迭代删除200个特征，保留3000个特征
        ##使用逻辑斯特回归分类器，具有coef_属性,在训练一个评估器后,coef_相当于每个特征的权重，RFE在每一次迭代中得到一个新的coef_
        #并删除权重低的特征，直到剩下的特征达到指定的数量
        #n_features_to_select选择的特征个数
        #这里的分类器可以选择其他的，比如svm
        raw_newdata=RFE(estimator=LogisticRegression(), n_features_to_select=num).fit_transform(raw_data, raw_label)
    
    #利用各个分类器去测试准确性
     #用SVM
        estimator = SVC(kernel='linear', C=1)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['SVM']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]
        
        #用KNN
        estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['KNN']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
        ##随机森林
        estimator=RandomForestClassifier(n_estimators=10)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['RF']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
        
        #AdaBoostClassifier
        estimator=AdaBoostClassifier(n_estimators=100)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['AdaBoost']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
        
        #朴素贝叶斯
        estimator=MultinomialNB(alpha=0.01)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['NB']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
        
        
         #神经网络
        estimator=MLPClassifier(alpha=1)
        ACC,nonDLB,DLB=statistics(raw_label,kFoldTest(estimator,raw_newdata,raw_label))
        df['MLP']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
        df.to_excel(writer,sheet_name='选择'+str(num)+'个特征的ACC',index=True)
    writer.save()
RFE_Acc([10,11,12,13,14,15,16,17,18,19,20],raw_data,raw_label)


################画出所选特征准确性对应的折线图##############
#将同一个表里不同单元格得到的ACC分别读到每一张表格里
filepath=r"E:\workspace\Dementia\RFE_select\二分类_select10-18_ACC.xlsx"
data=pd.read_excel(filepath,sheetname=[0,1,2,3,4,5,6,7,8,9,10])
for num in range(0,11):
    df=data[num]
    writer=pd.ExcelWriter(r'E:\workspace\Dementia\RFE_select\二分类RFE特征选择'+str(num+10)+'个的准确性.xlsx')
    df.to_excel(writer,index=True)
    writer.save()    


#将不同数目在不同分类器得到的ACC写到一张表格里
def writefile(input,output):    
    df_1 = pd.read_excel(filepath,sheetname=['选择10个特征的ACC','选择11个特征的ACC','选择12个特征的ACC','选择13个特征的ACC','选择14个特征的ACC','选择15个特征的ACC',\
                                  '选择16个特征的ACC','选择17个特征的ACC','选择18个特征的ACC','选择19个特征的ACC','选择20个特征的ACC'],columns=['SVM','KNN','RF','AdaBoost','NB','MLP'])   
    sy_clf =  pd.DataFrame(index=['选择10个特征的ACC','选择11个特征的ACC','选择12个特征的ACC','选择13个特征的ACC','选择14个特征的ACC','选择15个特征的ACC',\
                                  '选择16个特征的ACC','选择17个特征的ACC','选择18个特征的ACC','选择19个特征的ACC','选择20个特征的ACC'],columns=['SVM','KNN','RF','AdaBoost','NB','MLP'])
    for key in df_1:
        sy_clf.ix[key] = df_1[key].iloc[0]  #将ACC所在行的数据赋值给                     
        writer = pd.ExcelWriter(output)   
        sy_clf.to_excel(writer,'ACC准确性',index=True)
        writer.save()

inputfile=r"E:\workspace\Dementia\RFE_select\二分类_select10-18_ACC.xlsx"
outputfile=r"E:\workspace\Dementia\RFE_select\ACC_feature.xlsx"
writefile(inputfile,outputfile)  


def draw_line(data,title):
    X = np.arange(1,4)
    Y1 = list(data.iloc[0])    
    Y2 = list(data.iloc[1])
    Y3 = list(data.iloc[2])
    Y4 = list(data.iloc[3])
    Y5 = list(data.iloc[4])
    Y6 = list(data.iloc[5])
    fig = plt.figure()
    plt.xlabel('3 indicators')
    plt.ylabel('6 different classifier')
    plt.title(title)
    
    plt.plot(X,Y1,'b',marker='o',label='SVM')
    plt.plot(X,Y2,'r',marker='*',label='KNN')
    plt.plot(X,Y3,'c',marker='s',label='RF')
    plt.plot(X,Y4,'g',marker='d',label='AdaBoost')
    plt.plot(X,Y5,'y',marker='h',label='NB')
    plt.plot(X,Y6,'m',marker='H',label='MLP')
    plt.legend()
    plt.xticks(X,['ACC','DLB_ACC','nonDLB_ACC'])
    fig.savefig(r'/workspace/Dementia/RFE_select/'+'第'+title+'次特征ACC.jpg')


title = [10,11,12,13,14,15,16,17,18,19,20]
draw_1 = pd.read_excel(outputfile) #acc_真实1、2、3
draw_line(draw_1,title[0])
    














