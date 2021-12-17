#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as fisher_f
def SSE(y1, y2):
    # Sum of square of error
    s = 0
    for (a, b) in zip(y1, y2):
        temp = a-b
        temp = temp * temp
        s = s + temp
    return int(s)


# In[2]:


df = pd.read_csv('./GOOGL_weekly_return_volatility_detailed.csv')
year1 = df[df.Year == 2019]
year2 = df[df.Year == 2020]

year1 = year1[['Date','Week_Number', 'Close']]
year2 = year2[['Date', 'Week_Number', 'Close']]


# In[3]:


# cut data into month year1
monthPirceDic1 = { i:[] for i in range(1, 13)}

for (d, p) in zip(year1.Date, year1.Close):
    m = d[5:7]
    m = int(m)
    monthPirceDic1[m].append(p)


# cut data into month year2
monthPirceDic2 = { i:[] for i in range(1, 13)}

for (d, p) in zip(year2.Date, year2.Close):
    m = d[5:7]
    m = int(m)
    monthPirceDic2[m].append(p)

monthPirceDic1


# In[5]:


plt.plot(monthPirceDic1[1])


# # Count SSE for one regression function for both years

# In[34]:


# for year1
sse1 = []
for m in range(1, 13):
    yTest = monthPirceDic1[m]
    x = [i for i in range(len(yTest))]
    degree = 1
    weights = np.polyfit(x, yTest, degree)
    model = np.poly1d(weights)
    yPredict = model(x)
    sse1.append(SSE(yPredict, yTest))

# for year2
sse2 = []
for m in range(1, 13):
    yTest = monthPirceDic2[m]
    x = [i for i in range(len(yTest))]
    degree = 1
    weights = np.polyfit(x, yTest, degree)
    model = np.poly1d(weights)
    yPredict = model(x)
    sse2.append(SSE(yPredict, yTest))


print(sse1)
print(sse2)


# In[67]:


# find trend for year1
kSSEList1 = []
for m in range(1, 13):
    yTest = monthPirceDic1[m]
    kIndex = [i for i in range(1, len(yTest)-2)]
    degree = 1
    trend = []
    for k in kIndex:
        yTestF = yTest[0:k + 1]
        xf = [i for i in range(len(yTestF))]
        weightf = np.polyfit(xf, yTestF, degree)
        wf = weightf[0]
        model = np.poly1d(weightf)
        yPredictF = model(xf)



        yTestS = yTest[k+1:]
        xs = [i for i in range(len(yTestS))]
        weights = np.polyfit(xs, yTestS, degree)
        ws = weights[0]
        model = np.poly1d(weights)
        ypredictS = model(xs)

        if wf * ws < 0: # then it is a trend change
            sseTotal = SSE(yPredictF, yTestF) + SSE(ypredictS, yTestS)
            trend.append((k, sseTotal))

    trend = sorted(trend, key = lambda x: x[1])
    kSSEList1.append(trend[0])

# find trend for year2
kSSEList2 = []
for m in range(1, 13):
    yTest = monthPirceDic2[m]
    kIndex = [i for i in range(1, len(yTest)-2)]
    degree = 1
    trend = []
    for k in kIndex:
        yTestF = yTest[0:k + 1]
        xf = [i for i in range(len(yTestF))]
        weightf = np.polyfit(xf, yTestF, degree)
        wf = weightf[0]
        model = np.poly1d(weightf)
        yPredictF = model(xf)


        yTestS = yTest[k+1:]
        xs = [i for i in range(len(yTestS))]
        weights = np.polyfit(xs, yTestS, degree)
        ws = weights[0]
        model = np.poly1d(weights)
        ypredictS = model(xs)

        if wf * ws < 0: # then it is a trend change
            sseTotal = SSE(yPredictF, yTestF) + SSE(ypredictS, yTestS)
            trend.append((k, sseTotal))
           

    trend = sorted(trend, key = lambda x: x[1])
    try:
        kSSEList2.append(trend[0])
    except:
        kSSEList2.append('NA')

print(kSSEList1)
print()
print(kSSEList2)


# In[76]:


# compute f test for year 1
for i in range(12):
    l = sse1[i]
    l12 = kSSEList1[i][1]

    n = len(monthPirceDic1[i+1])

    F = (l - l12) / 2
    F = F * (n-4) / (l + l12)

    pValue = fisher_f.cdf(F, 2, n-4)

    print(pValue)

print()
# compute f test for year 2

for i in range(12):

    l = sse2[i]
    l12 = kSSEList2[i][1]
    
    n = len(monthPirceDic2[i+1])

    try:
        F = (l - l12) / 2
        F = F * (n-4) / (l + l12)

        pValue = fisher_f.cdf(F, 2, n-4)

        print(pValue)

    except:
        l12 = l
        F = (l - l12) / 2
        F = F * (n-4) / (l + l12)

        pValue = fisher_f.cdf(F, 2, n-4)

        print(pValue)


# In[ ]:




