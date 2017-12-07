#coding:utf-8  
from numpy import *  
 
import math  
"""
k_means
"""      
def loadDataSet(filename):  
    dataMat = []  
    fr = open(filename)  
    for line in fr.readlines():  
        element = line.strip().split(',')  
        number = []  
        for i in range(len(element)):  
            number.append(float(element[i]))  
        dataMat.append(number)  
    return dataMat  
#计算量元素欧式距离      
def distEclud(vecA, vecB):  
    count = len(vecA)  
    s = 0.0  
    for i in range(0, count):  
        s = s + power(vecA[i]-vecB[i], 2)  
    return sqrt(s)  
#找出离元素最近中心点的标号      
def clusterOfElement(means, element):  
    min_dist = distEclud(means[0], element)  
    lable = 0  
    for index in range(1, len(means)):  
        dist = distEclud(means[index], element)  
        if(dist < min_dist):  
            min_dist = dist  
            lable = index  
    return lable  

#计算每个簇的均值          
def getMean(cluster):   #cluster=[[[1,2],[1,2],[1,2]....],[[2,1],[2,1],[2,1],[2,1]...]]  
    num = len(cluster)  #1个簇的num，如上为3个  
    res = []  
    temp = 0  
    dim = len(cluster[0])  
    for i in range(0, dim):  
        for j in range(0, num):  
            temp = temp + cluster[j][i]  
        temp = temp / num  
        res.append(temp)  
    return res  
      
def kMeans():  
    k = 3  
    data = loadDataSet('data.txt')  
    print ("data is ", data)  
    inite_mean = [[1.1, 1], [1, 1],[1,2]]       
    count = 0  
    while(count < 1000):  
        count = count + 1  
        clusters = []  
        means = []  
        for i in range(k):  
            clusters.append([])  
            means.append([])  
        for index in range(len(data)):  
            lable = clusterOfElement(inite_mean, data[index])  
            clusters[lable].append(data[index])  
            
        for cluster_index in range(k):  
            mea = getMean(clusters[cluster_index])  
            for mean_dim in range(len(mea)):      
                means[cluster_index].append(mea[mean_dim])  
                
        for mm in range(len(means)):  
            for mmm in range(len(means[mm])): 
                #更新中心点
                inite_mean[mm][mmm] = means[mm][mmm]  
    print ("result cluster is ", clusters ) 
    print ("result means is ", inite_mean ) 
          
if __name__ == '__main__':
    kMeans()  