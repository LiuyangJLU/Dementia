# -*- coding: utf-8 -*-
"""

Python3
Required Packages
--pandas
--numpy
--sklearn
--scipy

Info
-name   :"liuyang"
-email  :'1410455465@qq.com'
-date   :2017-11-02
-Description 


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV#使用网格搜索进行调参



filepath = r'~/Dementia/Apr_1th_AD_VD_DLB/Q_AD_DLB_VD_after_normalizion_0_to_1.csv'
dataset = pd.read_csv(filepath)
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
            
raw_data = temp.iloc[:,2: ].drop('Diagnosis',1)#选择第二列以后的数据，并删除标签列
raw_label = list(temp.loc[:,'Diagnosis'])



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


def statistics(raw,test):
    VD_true = 0
    DLB_true = 0
    AD_true = 0
    for i in range(len(raw)):
        if raw[i] == 0 and test[i] == 0:
            DLB_true += 1
        if raw[i] == 1 and test[i] == 1:
            AD_true += 1
        if raw[i] == 2 and test[i] == 2:
            VD_true += 1
    ACC=(VD_true+AD_true+DLB_true)/len(raw)
    return ACC,VD_true,AD_true,DLB_true

def test_acc(raw_data,raw_label):
    writer = pd.ExcelWriter(r"E:\workspace\Dementia\ACC.xlsx") 
    df=pd.DataFrame(index=['ACC','DLB_ACC','AD_ACC','VD_ACC'])
    #利用各个分类器去测试准确性
     #用SVM
    estimator = SVC(kernel='linear', C=1)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['SVM']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]
    
    #用KNN
    estimator=neighbors.KNeighborsClassifier(n_neighbors=3)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['KNN']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    ##随机森林
    estimator=RandomForestClassifier(n_estimators=100)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['RF']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
     #DecisionTreeClassifier
    estimator=DecisionTreeClassifier(n_estimators=1000)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['DTree']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #AdaBoostClassifier
    estimator=AdaBoostClassifier(n_estimators=1000)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['AdaBoost']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    #朴素贝叶斯
    estimator=BernoulliNB(alpha=0.01)
    ACC,DLB,VD,AD=statistics(raw_label,kFoldTest(estimator,raw_data,raw_label))
    df['NB']=[ACC,DLB/DLB_count,VD/VD_count,AD/AD_count]#返回的每一列的值
    
    df.to_excel(writer,index=True)
    writer.save()
test_acc(raw_data,raw_label)



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(raw_data, raw_label)

x_train_2 = select_feature.transform(raw_data)
x_test_2 = select_feature.transform(raw_label)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")

# first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

        




sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)          
            
            
  

clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()          