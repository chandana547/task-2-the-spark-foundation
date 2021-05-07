#!/usr/bin/env python
# coding: utf-8

# # Task 2: Prediction Using Unsupervised ML
# Author : G Madhu chandana
# objective : From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# 1. Reading and Understanding the Data
# 

# # Step1: Importing Required Libraries

# In[1]:


# Importing Required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#Importing clustering libraries
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# # step 2: Reading the Dataset

# In[2]:


# Importing iris dataset

from sklearn import datasets
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data , columns = iris.feature_names)

iris_data.head()


# In[3]:


#Checking the shape of the dataset
iris_data.shape


# In[4]:


#Checking summary of the dataset
iris_data.info()


# # step 3: Defining function for Hopkins Score

# In[5]:


## Checking if the data is feasible for clustering

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# Checking Hopkins score for iris dataset and check whether it is feasible for clustering or not.

# In[6]:


# Using the function check hopkins statistic for the data
print(hopkins(iris_data))

We can proceed with clustering as the hokins statistic is good.
# # Step4: Rescaling the variables

# In[7]:


scaler = StandardScaler() #instantiate

#Fit_transform the data
iris_scaled = scaler.fit_transform(iris_data)

iris_scaled.shape


# Creating new dataframe with Scales variables

# In[9]:


#creating dataframe for the scaled variables
iris_scaled = pd.DataFrame(iris_scaled)
iris_scaled.columns = iris_data.columns
iris_scaled.head()


# # Step5: Modelling

# Find optimal number of customers
# 5.1 Silhouette Analysis

# In[10]:


range_n_clusters = [2,3,4,5,6,7,8]

for num in range_n_clusters:

  #initialise kmeans
  kmeans = KMeans(n_clusters = num, max_iter = 50 )
  kmeans.fit(iris_scaled)

  cluster_labels = kmeans.labels_

  #silhouette score
  silhouette_avg = silhouette_score(iris_scaled , cluster_labels)
  print("For n_clusters = {0}, the silhouette score is {1}".format(num , silhouette_avg))


#  5.2 Elbow Curve / SSD

# In[11]:


ssd = []
range_n_clusters = [2,3,4,5,6,7,8]
for num in range_n_clusters:
  kmeans = KMeans(n_clusters = num , max_iter = 50)
  kmeans.fit(iris_scaled)

  ssd.append(kmeans.inertia_)

#Plot SSD
plt.plot(ssd)


# In[12]:


#Final model with k = 3
kmeans = KMeans(n_clusters= 3, max_iter = 50)
kmeans.fit_predict(iris_scaled)


# In[13]:


kmeans.labels_


# In[14]:


#Assigning the values to dataframe
iris_data['cluster_labels'] = kmeans.labels_
iris_data.head()


# # step6: Visualization of model

#  6.1 Visualizing each column with Boxplot

# In[16]:


# Lets visualize clusters with every column

plt.figure(figsize = (10,16))

#Boxplot for Sepal length(cm)
plt.subplot(4,1,1)                          # creating subplots
plt.title('sepal length (cm)',fontsize=25)   # Adding and formatting title
sns.boxplot(x = iris_data['cluster_labels'], y=iris_data['sepal length (cm)'],palette='gist_heat',orient='v',fliersize=5)

#Boxplot for sepal width
plt.subplot(4,1,2)                         # creating subplots
plt.title('sepal width (cm)',fontsize=25)           # Adding and formatting title
sns.boxplot(x = iris_data['cluster_labels'], y = iris_data['sepal width (cm)'],palette='gist_heat',orient='v',fliersize=5)

#Boxplot for petal width
plt.subplot(4,1,3)                           # creating subplots
plt.title('petal length (cm)',fontsize=25)     # Adding and formatting title
sns.boxplot(x = iris_data['cluster_labels'], y = iris_data['petal length (cm)'],palette='gist_heat',orient='v',fliersize=5)

#Boxplot for petal length
plt.subplot(4,1,4)                           # creating subplots
plt.title('petal width (cm)',fontsize=25)     # Adding and formatting title
sns.boxplot(x = iris_data['cluster_labels'], y = iris_data['petal width (cm)'],palette='gist_heat',orient='v',fliersize=5)

plt.tight_layout()                     
plt.show()

Inferences :

Cluster 0 have larger petal width ,larger petal length, larger sepal length and average sepal width. So from this observation we can conclude that this species is Iris-versicolour.
Cluster 1 have average petal length, petal width and sepal length, lower sepal width . So from this observation we can conclude that this species is Iris-setosa.
Cluster 2 have lower petal width, petal length and sepal length ,larger sepal width. So from this observation we can conclude that this species is Iris-virginica.
# # Conclusion
# I was able to successfully carry-out prediction using Unsupervised Machine Learning.
# Thank You
