#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
def cutWeek(weekNumber, data):
    weekdata = []
    for i in range(weekNumber):
         temp = data[data.Week_Number == i]
         temp = temp.reset_index(drop=True)
         weekdata.append(temp)
    return weekdata

def labelMapping(year, week, label):
    labelMap = {}
    for (y, w, l) in zip(year, week, label):
        key = (y, w)
        value = l
        labelMap[key] = value
    return labelMap

def proficCalculator(data, fund):
    # Week 0 case
    week1Data = data[0]
    week1Label = week1Data.Label[0] # week 0 label

    if week1Label == 1:
        stock = True
        buyPrice = week1Data.Close[0] # week 0 first day price
        sellPrice = week1Data.Close[len(week1Data)-1] # week 0 last day price
    else:
        stock = False
        buyPrice = week1Data.Close[len(week1Data)-1] # week 0 last day price
        sellPrice = week1Data.Close[len(week1Data)-1] # week 0 last day price


    for df in data[1:]:
        nextWeekColor = df.Label[0]
        nextClosePrice = df.Close[len(df)-1]

        # stock + green = no action
        if (stock == True) and (nextWeekColor == 1):
            stock == True # Keep holding the stock
            buyPrice = buyPrice # Buy point stay
            sellPrice = nextClosePrice # Sell point move forward

        # stock + red = sell
        elif (stock == True) and (nextWeekColor == 0):
            r = 1 + (sellPrice - buyPrice) / sellPrice
            fund = fund * r
            buyPrice = nextClosePrice
            sellPrice = nextClosePrice
            stock = False
            
        # money + green = buy stock
        elif (stock == False) and (nextWeekColor == 1):
            buyPrice = buyPrice
            sellPrice = nextClosePrice
            stock = True
        # money + red = no action
        elif (stock == False) and (nextWeekColor == 0):
            buyPrice = nextClosePrice
            sellPrice = nextClosePrice
            stock = False

    # Last withdraw
    r = 1 + (sellPrice - buyPrice) / sellPrice
    fund = fund * r
    return fund


# In[37]:


dfLabel = pd.read_csv('./GOOGL_weekly_return_volatility.csv')
year1 = dfLabel[dfLabel.Year == 2019]
year2 = dfLabel[dfLabel.Year == 2020]


# # plot and provide best n, d

# In[38]:


xTrain = year1[['mean_return', 'volatility']]
yTrain = year1.label
xTest = year2[['mean_return', 'volatility']]
yTest = year2.label

nList = [1,3,5,7,9]
dList = [1,2,3,4,5]

errorMap = []

for n in nList:
    for d in dList:
        clf = RandomForestClassifier(max_depth=d, n_estimators=n, random_state=10)
        clf.fit(xTrain, yTrain)
        yPredict = clf.predict(xTest)
        errRate = np.mean(yPredict != yTest)
        errorMap.append((n, d, errRate))
errorMap = sorted(errorMap, key = lambda x:x[2])

# plot scatter
plt.figure(figsize=(12,8))
for (x, y, area) in errorMap:
    plt.scatter(x,y,s = area**5 * 100000, c='red', marker = 'o')
    plt.annotate('{:.2f}'.format(area), xy=(x, y), textcoords='offset points')

plt.title('ErrorRate')
plt.xlabel('N estimator')
plt.ylabel('Depth')


# In[39]:


errorMap


# # use optimal values to compute year 2

# In[40]:


n = 9
d = 3
clf = RandomForestClassifier(max_depth=d, n_estimators=n, random_state=10)
clf.fit(xTrain, yTrain)
yPredict = clf.predict(xTest)
print(accuracy_score(yTest, yPredict))

## Confusion Matrix I choose 
temp = confusion_matrix(yTest, yPredict)
print(temp)

tn = temp[0][0]
fn = temp[1][0]
tp = temp[1][1]
fp = temp[0][1]

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

print('TPR = {}, TNR = {}'.format(tpr, tnr))



# # Trading Strategy

# In[41]:


# Strategy check
dfDetail = pd.read_csv('./GOOGL_weekly_return_volatility_detailed.csv')
year2Detail = dfDetail[dfDetail.Year == 2020]
year2Detail = year2Detail.reset_index(drop = True)

## Add label to detail
lMap = labelMapping(year2.Year, year2.Week_Number, yPredict)
temp = []
for (y, w) in zip(year2Detail.Year, year2Detail.Week_Number):
    key = (y, w)
    temp.append(lMap[key]) 
year2Detail['Label'] = temp
year2Detail = year2Detail[['Year', 'Week_Number', 'Close', 'Label']]

## Cut goo2020
goo2020Week = cutWeek(53, year2Detail)




## trading 
total = proficCalculator(goo2020Week, 100)
print("Using Label: {}".format(total))

## trding BH
firstWeek = goo2020Week[0]
firstClose = firstWeek.Close[0]

lastWeek = goo2020Week[-1]
lastClose = lastWeek.Close[len(lastWeek)-1]

r = 1 + (lastClose - firstClose) / lastClose
total = 100 * r
print("Buy on first day and Sell on last day: {}".format(total))

