# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:02:25 2018
聚类分析   
@author: ly
"""

from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],
            X[:,1],
            c='white',
            marker='o',
            s=50)
plt.grid()
plt.show()


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

'''将簇数量设定为3个，指定先验的簇数量是K-mean算法的一个缺陷，设置
n_init=10,程序能够基于不同的随机初始中心点独立运行算法10次，并从中选择簇内误差平方和最小的作为最终模型，通过max_iter参数，
指定算法每轮运行的迭代次数，如果模型收敛，即使未达到预定迭代次数，算法也将终止，不过在k-means算法的某轮迭代中，可能会发生无法收敛的情况，
特别是当我们设置了较大的max_iter值时，更有可能产生此问题，解决收敛问题的一个方法是为tol参数设置一个较大的值，此参数能控制对簇内误差平方和的容忍度，
，'''