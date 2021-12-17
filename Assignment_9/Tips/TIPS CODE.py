#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[3]:


df = pd.read_csv("./tips.csv")
df


# # Average tip for lunch and for dinner

# In[4]:


# for dinner
dinDf = df[df.time == 'Dinner']
dinRate =  dinDf.tip.sum() / dinDf.total_bill.sum() 
print('dinner rate = {}'.format(dinRate))



# for lunch
lunDf = df[df.time == 'Lunch']
lunRate = lunDf.tip.sum() / lunDf.total_bill.sum()
print('lunch rate = {}'.format(lunRate))


# # Tip for each day

# In[5]:


# Fri
data = df[df.day == 'Fri']
rate = data.tip.sum() / data.total_bill.sum()
print('Fri rate = {}'.format(rate))
# Sat
data = df[df.day == 'Sat']
rate = data.tip.sum() / data.total_bill.sum()
print('Sat rate = {}'.format(rate))
# Sun
data = df[df.day == 'Sun']
rate = data.tip.sum() / data.total_bill.sum()
print('Sun rate = {}'.format(rate))
# Thur
data = df[df.day == 'Thur']
rate = data.tip.sum() / data.total_bill.sum()
print('Thur rate = {}'.format(rate))


# # Highest tip (which day and time)
# 

# In[6]:


# Fri
data = df[(df.day == 'Fri')]
data = data[(data.time == 'Lunch')]
rate = data.tip.sum() / data.total_bill.sum()
print('Fri lunch rate = {}'.format(rate))

data = df[(df.day == 'Fri')]
data = data[(data.time == 'Dinner')]
rate = data.tip.sum() / data.total_bill.sum()
print('Fri dinner rate = {}'.format(rate))


# Sat
data = df[(df.day == 'Sat')]
data = data[(data.time == 'Lunch')]
rate = 0
print('Sat lunch rate = {}'.format(rate))

data = df[(df.day == 'Sat')]
data = data[(data.time == 'Dinner')]
rate = data.tip.sum() / data.total_bill.sum()
print('Sat dinner rate = {}'.format(rate))

# Sun
data = df[(df.day == 'Sun')]
data = data[(data.time == 'Lunch')]
rate = 0
print('Sun lunch rate = {}'.format(rate))

data = df[(df.day == 'Sun')]
data = data[(data.time == 'Dinner')]
rate = data.tip.sum() / data.total_bill.sum()
print('Sun dinner rate = {}'.format(rate))


# Thur
data = df[(df.day == 'Thur')]
data = data[(data.time == 'Lunch')]
rate = data.tip.sum() / data.total_bill.sum()
print('Thur lunch rate = {}'.format(rate))

data = df[(df.day == 'Thur')]
data = data[(data.time == 'Dinner')]
rate = data.tip.sum() / data.total_bill.sum()
print('Thur dinner rate = {}'.format(rate))


# # Correlation between meal price and tips

# In[7]:


mealPrice = df.total_bill
tip = df.tip
pearsonR,_ = pearsonr(mealPrice, tip)

print('correlation is {}'.format(pearsonR))


# # Correlation between size and tips

# In[8]:


size = df.loc[:, 'size']
tip = df.tip
pearsonR,_ = pearsonr(size, tip)

print('correlation is {}'.format(pearsonR))


# # Percentage of people are smoking

# In[9]:


smokeNO = df[df.smoker == "No"].smoker.count()
smokeYes = df[df.smoker == 'Yes'].smoker.count()
rate = smokeYes / (smokeNO + smokeYes)
print('smoke rate = {}'.format(rate))


# # Tips increasing with time in each day?

# In[10]:


# Fri
data = df[df.day == 'Fri']
data = list(data.tip)

n = len(data)
temp = [i for i in range(n)]


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(temp, data)
plt.title('Fri')


# In[11]:


# Sat
data = df[df.day == 'Sat']
data = list(data.tip)

n = len(data)
temp = [i for i in range(n)]


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(temp, data)
plt.title('Sat')


# In[12]:


# Sun
data = df[df.day == 'Sun']
data = list(data.tip)

n = len(data)
temp = [i for i in range(n)]


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(temp, data)
plt.title('Sun')


# In[13]:


# Thur
data = df[df.day == 'Thur']
data = list(data.tip)

n = len(data)
temp = [i for i in range(n)]


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(temp, data)
plt.title('Thur')


# # Difference in correlation between tip amounts from smokers and non-smokers

# In[14]:


smokeDf = df[df.smoker == 'Yes']
smokeTip = smokeDf.tip
y = [0 for i in range(len(smokeTip))]
plt.scatter(smokeTip, y, color = 'red', label='smoke')

nonSmokeDf = df[df.smoker == 'No']
nonSmokeTip = nonSmokeDf.tip
y = [0.1 for i in range(len(nonSmokeTip))]
plt.scatter(nonSmokeTip, y, color = 'blue', label='non smoke')

plt.legend()


# In[15]:


plt.hist(smokeTip, color = 'red', label = 'smoke')

plt.hist(nonSmokeTip, color = 'Blue', alpha = 0.5, label = 'non smoke')


plt.legend()


# In[42]:


print(smokeTip.var())
print(nonSmokeTip.var())


# In[34]:


import scipy.stats as st

def f_test(x, y, alt="two_sided"):
    """
    Calculates the F-test.
    :param x: The first group of data
    :param y: The second group of data
    :param alt: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a tuple with the F statistic value and the p-value.
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var() / y.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0*(1.0 - st.f.cdf(f, df1, df2))
    return f, p


# In[43]:


f_test(smokeTip, nonSmokeTip)


# In[ ]:




