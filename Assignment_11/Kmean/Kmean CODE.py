#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

from sklearn.cluster import *


# In[12]:


df = pd.read_csv('./GOOGL_weekly_return_volatility.csv')
feature = df[['mean_return', 'volatility']].values
labelTrue = df.label


# In[13]:


df


# In[14]:


# plot data point
plt.scatter(x = df[df.label == 1].mean_return, y = df[df.label==1].volatility, color='green')
plt.scatter(x = df[df.label == 0].mean_return, y = df[df.label==0].volatility, color='red')


# # Knee method

# In[15]:


inList = []
kList = [i+1 for i in range(8)]
for k in kList:
    
    clf = KMeans(n_clusters=k, random_state=0)
    yPre = clf.fit_predict(feature)
    inList.append(clf.inertia_)


# In[16]:


plt.plot(kList, inList,marker='o')


# # Optimal K

# In[18]:


k = 2
clf = KMeans(n_clusters=2, random_state=0)
yPre = clf.fit_predict(feature)
df['labelPredict'] = yPre


# In[19]:


df


# In[22]:


cluster1 = df[df.labelPredict == 0]
cluster2 = df[df.labelPredict == 1]

# cluster 1 
greenRate = sum(cluster1.label) / len(cluster1)
redRate = 1 - greenRate
print(greenRate, redRate)


# cluster 2 
greenRate = sum(cluster2.label) / len(cluster2)
redRate = 1 - greenRate
print(greenRate, redRate)


# In[ ]:




