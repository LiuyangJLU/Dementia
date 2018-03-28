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

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score



filepath=r"E:\workspace\Dementia\Q_DLB_nonDLB_after_removing_education_normalizion_0_to_1.csv"
temp=pd.DataFrame.from_csv(filepath, index_col=None)



 ##加载类标签，并对每个类别进行计数
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
        
#特征和类标           
raw_data=temp.drop('Diagnosis',1)
raw_labels=list(temp.loc[:,'Diagnosis'])


#交叉验证均方误差的函数，均方根误差是用来衡量观测值同真值之间的偏差
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, raw_data, raw_labels, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)

#这个参数α的选择是通过交叉验证得到
model_ridge = Ridge()   
alphas = [0.05, 0.1, 0.3,1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()#ridge误差率

#利用lassoCV进行交叉验证得到最优权重
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(raw_data, raw_labels)
#利用ridgeCV进行交叉验证得到最优权重
model_ridge = RidgeCV(alphas = [0.05, 0.1, 0.3,1, 3, 5, 10, 15, 30, 50, 75]).fit(raw_data,raw_labels)

rmse_cv(model_lasso).mean()#0.53420332961556849
rmse_cv(model_ridge).mean()#0.53366039411844268


coef_lasso = list(model_lasso.coef_)#通过lasso得到的特征权重
coef_ridge = list(model_ridge.coef_)
feature_list=list(raw_data.columns)

#coef = pd.DataFrame(model_lasso.coef_, index = raw_data.columns)#查看各特征的系数
feature_coef = pd.DataFrame({'feature':feature_list,'coef':coef_lasso})
feature_coef = feature_coef.sort_index(by='coef',ascending=False)    
#将特征权重结果写入Excel
writer = pd.ExcelWriter(r'E:\workspace\1110test\loss_coef_descend.xlsx')
feature_coef.to_excel(writer,index=False)        
writer.save() 

#####画出各特征的权重直线图###################
print("Lasso picked " + str(sum(coef_lasso != 0)) + " variables and eliminated the other " +  str(sum(coef_lasso == 0)) + " variables")

"""
Lasso picked 38 variables and eliminated the other 15 variables
"""

#选择特征排名前10和后10的特征系数
imp_coef = pd.concat([feature_coef.sort_values( by='coef',ascending =False).head(10),
                     feature_coef.sort_values(by='coef',ascending =False).tail(10)])
imp_coef.shape#(20,2)
#for i in len(feature_list):
#    import_coef = pd.DataFrame({'feature':feature_list[20],'imp_coef':imp_coef})





#lasso回归系数的面积展示
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")



#let's look at the residuals as well:
plt.rcParams['figure.figsize'] = (6.0, 6.0)
Predict = pd.DataFrame({"Predict":model_lasso.predict(raw_data), "true":raw_labels})
Predict["residuals"] = Predict["true"] - Predict["Predict"]
Predict.plot(x = r"Predict", y = r"residuals",kind = "scatter")


#原始数据和原始标签
coef_Index=imp_coef['feature']#选择权重排名前10的特征
dataset = raw_data.loc[:,coef_Index].as_matrix(columns=None) #返回的不是numpy矩阵，而是numpy-array     
label = list(temp.loc[:,'Diagnosis'])

def Access(raw,test):
    nonDLB_true=0
    DLB_true=0
    for i in range(len(raw)):
        if raw[i]==0 and test[i]==0:
            nonDLB_true+=1
        if raw[i]==1 and test[i]==1:
            DLB_true+=1
    ACC=(DLB_true+nonDLB_true)/len(raw)
    return ACC,nonDLB_true,DLB_true 


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


def main(dataset,label):
    writer=pd.ExcelWriter(r"E:\workspace\1110test\binary_qian10hou10.xlsx")
    df=pd.DataFrame(index=['ACC','DLB_ACC','nonDLB_ACC'])
        
    estimator=SVC(kernel='rbf',C=10)#C值=10效果最好
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_rbf'] = [ACC,nonDLB/nonDLB_count,DLB/DLB_count]


    #用KNN
    estimator=SVC(kernel='poly',C=10)
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_poly'] =[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    ##随机森林
    estimator = RandomForestClassifier(n_estimators=10)
    ACC,nonDLB,DLB = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_linear'] =[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值\
    
    estimator=SVC(kernel='sigmoid',C=10)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_sigmoid']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]
    
    #RF
    estimator=RandomForestClassifier(n_estimators=100,criterion="entropy")
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['RF']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]
    
     #用KNN
    
    estimator = neighbors.KNeighborsClassifier(n_neighbors=10)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['KNN']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
  #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['AdaBoost']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=BernoulliNB(alpha=0.01)
    ACC,nonDLB,DLB=Access(label,kFoldTest(estimator,dataset,label))
    df['NB']=[ACC,nonDLB/nonDLB_count,DLB/DLB_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
    
    
main(dataset,label)