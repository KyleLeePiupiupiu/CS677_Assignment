#!/usr/bin/env python
# coding: utf-8

# # Question1

# In[129]:


import pandas as pd
import numpy as np


# In[131]:


goo = pd.read_csv('./GOOGL.csv')
spy = pd.read_csv('./spy.csv')
goo = goo[['Year','Month', 'Weekday', 'Return']]


# In[45]:


def cutYear(data):
    temp = []
    for i in range(5):
        y = 2016 + i
        temp.append(data[data.Year == y])
    return temp

def cutWeek(data):
    m = data[data.Weekday == 'Monday']
    t = data[data.Weekday == 'Tuesday']
    w = data[data.Weekday == 'Wednesday']
    r = data[data.Weekday == 'Thursday']
    f = data[data.Weekday == 'Friday']
    p(m.Return)
    p(t.Return)
    p(w.Return)
    p(r.Return)
    p(f.Return)
    print()
    
def p(data):
    print(len(data), data.mean(), data.std())
    


# In[46]:


# R
temp = cutYear(goo)
for year in temp:
    cutWeek(year)
    


# In[47]:


# R+
temp = cutYear(goo[goo.Return >= 0])
for year in temp:
    cutWeek(year)


# In[48]:


# R-
temp = cutYear(goo[goo.Return < 0])
for year in temp:
    cutWeek(year)


# # Question3

# In[49]:


cutWeek(goo)
cutWeek(spy)


# In[50]:


# normal distribution
## GOO
mu = goo.Return.mean()
sig = goo.Return.std()
interval = (mu - 2 * sig, mu + 2 * sig)
totalDay = len(goo)
upOutDay = len(goo[goo.Return > interval[1]])
downOutDay = len(goo[goo.Return < interval[0]])
print((upOutDay + downOutDay) / totalDay)

## SPY
mu = spy.Return.mean()
sig = spy.Return.std()
interval = (mu - 2 * sig, mu + 2 * sig)
totalDay = len(spy)
upOutDay = len(spy[spy.Return > interval[1]])
downOutDay = len(spy[spy.Return < interval[0]])
print((upOutDay + downOutDay) / totalDay)


# # Question4

# In[51]:


# GOO
d = goo
d = d[d.Return >= 0]
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)

# SPY
d = spy
d = d[d.Return >= 0]
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)


# In[52]:


# BU id, mine is 94
## GOO
d = goo
d = d[d.Return >= 0]
r = d.Return
totalFund = 100
day = 0
for rr in r:
    totalFund = totalFund * (1 + rr)
    if totalFund >= 194:
        break
    day = day + 1
print(day)

## SPY
d = spy
d = d[d.Return >= 0]
r = d.Return
totalFund = 100
day = 0
for rr in r:
    totalFund = totalFund * (1 + rr)
    if totalFund >= 194:
        break
    day = day + 1
print(day)


# # Question5

# In[53]:


# BH
## GOO
d = goo
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)

## SPY
d = spy
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)


# In[73]:


# Summer Vacation
## GOO
d = goo
d = d[(d.Month != 6) & (d.Month != 7) & (d.Month != 8)]
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)


## SPY
d = spy
d = d[(d.Month != 6) & (d.Month != 7) & (d.Month != 8)]
r = d.Return
totalFund = 100
for rr in r:
    totalFund = totalFund * (1 + rr)
print(totalFund)


# In[76]:


# Vacation on each month
## GOO
for month in range(12):
    month = month + 1
    d = goo
    d = d[d.Month != month]
    r = d.Return
    totalFund = 100
    for rr in r:
        totalFund = totalFund * (1 + rr)
    print(totalFund)

print()
## SPY
for month in range(12):
    month = month + 1
    d = goo
    d = d[d.Month != month]
    r = d.Return
    totalFund = 100
    for rr in r:
        totalFund = totalFund * (1 + rr)
    print(totalFund)



# # Quesiton6

# In[105]:


import random
random.seed(1)
def oracle(data, p):
    orableLable = []
    for i in range(len(data)):
        r = random.uniform(0,1)
        if p >= r:
            orableLable.append(1)
        else:
            orableLable.append(0)
    
    return orableLable


# In[108]:


pList = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# GOO
d = goo
for p in pList:
    oLabel = oracle(d, p)
    totalR = 1
    fund = 100
    for (r, o) in zip(d.Return, oLabel):
        if (r >= 0) & (o == 1):
            totalR = totalR * (1 + r)
        elif (r >= 0) & (o == 0):
            continue
        elif (r <= 0) & (o == 1):
            continue
        elif (r <= 0) & (o == 0):
            totalR = totalR * (1 + r)
       
    print('p = {}, return = {}'.format(p, fund * totalR))
print()
# SPY
d = spy
for p in pList:
    oLabel = oracle(d, p)
    totalR = 1
    fund = 100
    for (r, o) in zip(d.Return, oLabel):
        if (r >= 0) & (o == 1):
            totalR = totalR * (1 + r)
        elif (r >= 0) & (o == 0):
            continue
        elif (r <= 0) & (o == 1):
            continue
        elif (r <= 0) & (o == 0):
            totalR = totalR * (1 + r)
       
    print('p = {}, return = {}'.format(p, fund * totalR))


# # Quesiton7

# In[119]:


# best 10
## GOO
d = goo
d = d[d.Return >= 0].Return
t = d.nlargest(10).index
d = d.drop(index=t)
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
print(100 * totalR)

## SPY
d = spy
d = d[d.Return >= 0].Return
t = d.nlargest(10).index
d = d.drop(index=t)
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
print(100 * totalR)


# In[122]:


# Worst 10
## GOO
d = goo
t = d.Return.nsmallest(10)
d = d[d.Return >= 0].Return
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
for r in t:
    totalR = totalR * (1 + r)
print(100 * totalR)

## SPY
d = spy
t = d.Return.nsmallest(10)
d = d[d.Return >= 0].Return
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
for r in t:
    totalR = totalR * (1 + r)
print(100 * totalR)


# In[128]:


# best 5 worst 5 
## GOO
d = goo
t = d.Return.nsmallest(5)
d = d[d.Return >= 0].Return
i = d.nlargest(5).index
d = d.drop(index=i)
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
for r in t:
    totalR = totalR * (1 + r)
print(100 * totalR)

## SPY
d = spy
t = d.Return.nsmallest(5)
d = d[d.Return >= 0].Return
i = d.nlargest(5).index
d = d.drop(index=i)
totalR = 1
for r in d:
    totalR = totalR * (1 + r)
for r in t:
    totalR = totalR * (1 + r)
print(100 * totalR)

