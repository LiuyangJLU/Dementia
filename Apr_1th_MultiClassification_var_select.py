# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:22:53 2017
Python3
Required Packages
--pandas
--numpy
--sklearn
--scipy

Info
-name   :"liuyang"
-email  :'1410455465@qq.com'
-date   :2017-10-16
-Description 
    Dementia Apr_1th_AD_VD_DLB MultiClassification  三分类
    方差排序，将方差大于0.1,0.15，0.17，0，2,0.22的特征挑选出来，利用交叉验证，
    查看各个分类器的正确率，并输出各个分类器的正确率到表格
    cross_validation
    
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
    


def var_select_ACC(var_value,data):
    "传入一个值，将方差大于该值的特征提取出来，进行训练，用交叉验证评估模型的准确率，将各个准确率写入excel表格"
    columns_var_descend=temp.var().sort_values(ascending=False)#求样本方差，按降序排列
    writer=pd.ExcelWriter(r"E:\workspace\Dementia\var_select\var_descend.xlsx")#每列方差降序写入excel表格中
    pd.DataFrame(columns_var_descend).to_excel(writer,index=True)
    writer.save()
    
    ##加载类标签，并对每个类别进行计数
    VD_count=0
    AD_count=0
    DLB_count=0  
    for i in range(len(data)):
        if data.loc[i,'Diagnosis'] =='DLB' :
            data.loc[i,'Diagnosis']=0
            DLB_count+=1
        elif data.loc[i,'Diagnosis'] =='AD' :            
            data.loc[i,'Diagnosis']=1
            AD_count+=1
        else :
            data.loc[i,'Diagnosis']=2
            VD_count+=1     
    
    writer=pd.ExcelWriter(r"E:\workspace\Dementia\var_select\var_select_ACC.xlsx")#要读入的文件位置
    
    
    
#剔除小于指定方差值要求的列    
    for temp_var in var_value:
        temp_data=data.copy()#把原始数据copy到临时变量里
        i=-1
        while(columns_var_descend[i]<temp_var):#当所在列的方差值小于指定方差值，删除所在列
            del temp_data[columns_var_descend.index[i]]
            i-=1
        raw_data=temp_data.drop('Diagnosis',1).as_matrix(columns=None) #原始数据
        target_data=list(temp.ix[:,'Diagnosis'])#原始标签
    
    
    #采用 K-Fold 交叉验证 得到 acc 
        def kFoldTest(estimator):
            predict=[]#将得到的结果放到列表里
            kf = KFold(n_splits=10)#10折交叉验证    
            for train_index, test_index in kf.split(raw_data):#kf.split(X)返回的是原始数据中进行分裂后train和test的索引值
#        print("TRAIN:", train_index, "TEST:", test_index)#查看如何分割数据
                X_train, X_test = raw_data[[train_index]], raw_data[[test_index]]#原始数据的训练集和测试集
        #Y_test在这里没作用，为了数据变量对齐0.0
                Y_train, Y_test = target_data[:test_index[0]]+target_data[test_index[-1]+1:], target_data[test_index[0]:test_index[-1]+1]        
                estimator.fit(X_train,Y_train)#分类器对每个训练集进行训练
                test_target_temp=estimator.predict(X_test)#分类器在测试集中去测试
                predict.append(test_target_temp)#结果返回到列表中
            test_target = [i for dataset in predict for i in dataset]#将10次测试集展平
            return test_target
        

    
        #返回准确率，统计正确分类的数目
        def statistics(raw,target):
            DLB_true=0#DLB标签正确分类初始记为0
            AD_true=0#AD标签正确分类初始为0
            VD_true=0#VD标签正确分类初始为0
            for i in range(len(raw)):#在每个原始数据里做for循环
                if raw[i]==0 and target[i]==0:#如果原始数据和分类后的值同时为0
                    DLB_true+=1#计数
                if raw[i]==1 and target[i]==1:
                    AD_true+=1
                if raw[i]==2 and target[i]==2:
                    VD_true+=1
            ACC=(DLB_true+AD_true+VD_true)/len(raw)#返回的是预测的正确率
            return ACC,DLB_true,AD_true,VD_true   #最终返回几个评价指标
    
        df=pd.DataFrame(index=['ACC','DLB_ACC','VD_ACC','AD_ACC'])
    
        ##svm
        estimator = SVC(kernel='linear', C=1)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['SVM']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    
        #用KNN
        estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
        
        ##随机森林
        estimator=RandomForestClassifier(n_estimators=10)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
        #AdaBoostClassifier
        estimator=AdaBoostClassifier(n_estimators=100)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
        #朴素贝叶斯
        estimator=MultinomialNB(alpha=0.01)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]


        #神经网络
        estimator=MLPClassifier(alpha=1)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['MLP']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
        #决策树
        estimator=DecisionTreeClassifier(random_state=0)
        ACC,DLB,VD,AD=statistics(target_data,kFoldTest(estimator))
        df['DTree']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
        df['特征数']=len(temp_data.columns)-1
        df.to_excel(writer,sheet_name='方差大于'+str(temp_var),index=True)
    writer.save()
var_select_ACC([0.1,0.15,0.17,0.2,0.22],temp)#调用函数

    
#==============================================================================
#     #加载类标签
#     def load_class(dataset):  
#     class_set = pd.read_csv(dataset,index_col = 0)
#     labels = class_set["Diagnosis"]#类标签为'Diagnosis'
#     result = []
#     def convert(label):
#         if label == 'VD':           
#             result.append(0)          
#         if label == 'AD':
#             result.append(1)           
#         if label == 'DLB':
#             result.append(2)           
#     labels.apply(func = convert)#将convert函数应用到labels上       
#     return np.array(result)
#==============================================================================
#==============================================================================
# temp_var_descend=dataset.var().sort_values(ascending=False)#求方差进行降序排列
#==============================================================================
#==============================================================================
# writer=pd.ExcelWriter(r'E:\workspace\Dementia\temp_var_descend.xlsx')
# pd.DataFrame(temp_var_descend).to_excel(writer,index=True)
# writer.save()
#==============================================================================


###根据方差值进行筛选#############
#根据传进的一个特征的方差，筛选冗余特征
#==============================================================================
# def var_select(var_value,dataset):
#     "传入一个参数，去排除方差比某个值小的特征"
#     data=pd.read_csv(filename)
#     temp=data.copy()
#     result=[]
#     for i in range(len(temp)):
#         flag=False
#         for j in range(min(temp['Diagnosis']-1)):
#             strtemp=temp.ix[i,j]
#             temp_var=strtemp.var()#求方差
#             if temp_var[:j]<0.15:
#                 flag=True
#             if flag:
#                 result=temp_var[:j].drop(i)
#     return np.arry(result)
#             elif temp_var[0:j]<0.1:
#==============================================================================
                
                
        
    


#函数:求方差阈值的选择，读入excel
#==============================================================================
# def varsize(filename):
#     data=pd.read_csv(filename)
#     var_value=data.var().sort_values(ascending=False)
#     i=-1
#     if var_value[i]<0.15:
#         del data[var_value,index[i]]
#     else:
#        var_value[i]
#     for i in range(temp_value[i]):
#     if var_value[i]<0.15:
#==============================================================================
