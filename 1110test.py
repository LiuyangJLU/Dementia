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
from sklearn.model_selection import GridSearchCV


filepath=r"E:\workspace\Dementia\Q_AD_DLB_VD_after_normalizion_0_to_1.csv"
temp=pd.DataFrame.from_csv(filepath, index_col=None)



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
#特征和类标           
raw_data=temp.drop('Diagnosis',1)
raw_labels=list(temp.loc[:,'Diagnosis'])
 

#交叉验证均方误差的函数，均方根误差是用来衡量观测值同真值之间的偏差
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, raw_data, raw_labels, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)


#利用ridgeCV
model_ridge = Ridge()   
alphas = [0.05, 0.1, 0.3,1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()#ridge误差率


rmse_cv(model_ridge).mean()
#利用lassoCV进行交叉验证得到最优权重
model_lasso = Lasso()
alphas = [1,0.1,0.01,0.001,0.0001,0]
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean()
            for alpha in alphas]
cv_lasso =pd.Series(cv_lasso,index = alphas)
cv_lasso.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_lasso.min()



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(raw_data,raw_labels)
rmse_cv(model_lasso).mean()


coef_lasso = list(model_lasso.coef_)#通过lasso得到的特征权重
feature_list=list(raw_data.columns)

#coef = pd.DataFrame(model_lasso.coef_, index = raw_data.columns)#查看各特征的系数
feature_coef = pd.DataFrame({'feature':feature_list,'coef':coef_lasso})
feature_coef = feature_coef.sort_index(by='coef',ascending=False)    
#将特征权重结果写入Excel
writer = pd.ExcelWriter(r'E:\workspace\1110test\coef_descend.xlsx')
feature_coef.to_excel(writer,index=False)        
writer.save() 

#####画出各特征的权重直线图###################
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

"""
Lasso picked 38 variables and eliminated the other 15 variables
"""

#

#选择特征排名前10和后10的特征系数
imp_coef = pd.concat([feature_coef.sort_values(by='coef',ascending =False).head(15),
                     feature_coef.sort_values(by='coef',ascending =False).tail(15)])
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
    writer=pd.ExcelWriter(r"E:\workspace\1110test\qian15hou15.xlsx")
    df=pd.DataFrame(index=['ACC','DLB_ACC','VD_ACC','AD_ACC'])
        
    estimator=SVC(kernel='rbf',C=10)#C值=10效果最好
    ACC,DLB,VD,AD = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_rbf'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]


    #用KNN
    estimator=SVC(kernel='poly',C=10)
    ACC,DLB,VD,AD = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_poly'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    ##随机森林
    estimator = RandomForestClassifier(n_estimators=10)
    ACC,DLB,VD,AD = Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_linear'] = [ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值\
    
    estimator=SVC(kernel='sigmoid',C=10)
    ACC,DLB,VD,AD=Access(label,kFoldTest(estimator,dataset,label))
    df['SVM_sigmoid']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #RF
    estimator=RandomForestClassifier(n_estimators=100,criterion="gini")
    ACC,DLB,VD,AD=Access(label,kFoldTest(estimator,dataset,label))
    df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
     #用KNN
    
    estimator = neighbors.KNeighborsClassifier(n_neighbors=10)
    ACC,DLB,VD,AD=Access(label,kFoldTest(estimator,dataset,label))
    df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
  #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=100)
    ACC,DLB,VD,AD=Access(label,kFoldTest(estimator,dataset,label))
    df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=BernoulliNB(alpha=0.01)
    ACC,DLB,VD,AD=Access(label,kFoldTest(estimator,dataset,label))
    df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
    
    
main(dataset,label) 
#选择分类器
def select_estimator(case):

    if case == "SVM":
        estimator = SVC()
    elif case == "KNN":
        estimator = KNeighborsClassifier()
    elif case == "DT":
        estimator = DecisionTreeClassifier()
    elif case == "NB":
        estimator = BernoulliNB()
    elif case == "LG":
        estimator = LogisticRegression() 
    elif case == "LSVM":
        estimator = LinearSVC()  
    elif case == "AdaBoost":
        estimator = AdaBoostClassifier()
    return estimator



#分类准确性返回混淆矩阵    
def get_acc(estimator,X,y):
    scores = []
    kf = KFold(n_splits=10)
    cm = np.zeros((3,3))    
    for train_index,test_index in kf.split(X,y):
        X_train,X_test = dataset[train_index],dataset[test_index]
        print(np.shape(X_train),np.shape(X_test))#方便查看输出
        #列表的切片访问，如果第一个索引是0 ，可以省略
        Y_train,Y_test = labels[:test_index[0]+test_index[-1]+1:],labels[test_index[0]:test_index[-1]+1]
        print(np.shape(Y_train),np.shape(Y_test))#方便查看输出
        estimator.fit(X_train,Y_train)
        score.append(estimator.score(X_test,Y_test))
        cm+=confusion_matrix(Y_test,estimator.predict(X_test))
    return np.mean(scores),cm 



#三分类的准确性   
def three_class_acc(dataset,labels,estimator_list,kf):
    max_estimator_aac = 0
    for estimator in estimator_list:
        estimator_aac,cm = get_acc(OneVsOneClassifier(select_estimator(estimator)),dataset,labels,kf)
        #print("the acc for {}: {}".format(estimator,estimator_aac))
        if estimator_aac > max_estimator_aac:
            max_estimator_aac = estimator_aac   #记录对于 k 个 特征 用五个estimator 得到的最大值
    #print("-"*20)        
    #print("the macc is: {}\n".format(max_estimator_aac))
    #print(cm)  
    #print("\n")      
    return max_estimator_aac







#classes分类标准
def single(dataset_filename,label,alpha = 0.01,classes = [[1],[2,3]]):

    dataset,labels,feature_list = prepare(dataset_filename,json_filename,classes,alpha)
    
    scores = test_acc(dataset,labels)
    print("for classes {}, the acc is: {}".format(classes,scores))

    feature_list = dataset.columns.tolist()
    feature_names = get_name(name_index_dic,feature_list)

    """
    print("the dataset shape is(samples,features): {}".format(str(dataset.shape)))
    print("-"*30)

    print("for different classes: {}\n".format(str(classes)))
    print("the features name is: ")
    print(feature_names)

    
    with open("result.txt","a") as f:
        f.write("for classes: {}\n".format(str(classes)))
        f.write("the dataset shape is(samples,features): {}\n".format(str(dataset.shape)))
        f.write("the acc is: {}\n".format(scores))
        f.write("the feature name is: {}\n\n".format(str(feature_names)))
    """
    return feature_names  


feature_list=temp.drop('Diagnosis',1).columns.tolist()   

for  num in range(raw_data.shape[1]):
    
def load_label(labels):
    labels_list = [[[1],[2]],[[1],[3]],[[2],[3]],[[1],[2,3]],[[2],[1,3]],[[3],[1,2]]]#将三分类的问题转换为二分类的问题
    """对标签进行分类，"""
    for labels in labels_list:
        


def kFoldTest(estimator,raw_data,raw_label):
    predict=[]
    kf=KFold(n_splits=10)   
    #for循环主要是把数据索引
    for train_index,test_index in kf.split(raw_data):
        X_train,X_test = raw_data[train_index],raw_data[test_index]#将数据分为验证集和测试集
        #将标签分为验证集和测试集，这里对类别的总数是从索引0到最后一个元素(index[-1]+1)的和加起来。
        #Y_test这个标签没用，只是做对称
        Y_train, Y_test = raw_label[:test_index[0]]+raw_label[test_index[-1]+1:], raw_label[test_index[0]:test_index[-1]+1] 
        estimator.fit(X_train, Y_train)
        test_target_temp=estimator.predict(X_test)
        predict.append(test_target_temp)
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target

def Access(raw,test):
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