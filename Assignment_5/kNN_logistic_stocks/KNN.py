#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('./GOOGL_weekly_return_volatility.csv')
year1 = df[df.Year == 2019]
year2 = df[df.Year == 2020]


# In[3]:


# year1 knn accuracy
kList = [3,5,7,9,11]
accuracy = []
x = year1[['mean_return', 'volatility']]
y = year1.label
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.4, random_state=0)
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xTrain, yTrain)
    yPredict = knn.predict(xTest)
    accuracy.append(accuracy_score(yTest, yPredict))

plt.plot(kList, accuracy)
print(accuracy)


# In[87]:


# Optimal k is 3,9,11
kList = [3,9,11]
accuracy = []
x = year1[['mean_return', 'volatility']]
y = year1.label
xTest = year2[['mean_return', 'volatility']]
yTest = year2.label
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y)
    yPredict = knn.predict(xTest)
    print('k = {}, accuracy = {}'.format(k, accuracy_score(yTest, yPredict)))


# Confusion Matrix I choose k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain, yTrain)
labelYear2 = knn.predict(xTest)

temp = confusion_matrix(yTest, labelYear2)
print(temp)

tpr = temp[0][0] / (temp[0][0] + temp[0][1])
tnr = temp[1][1] / (temp[0][1] + temp[1][1])
print('TPR = {}, TNR = {}'.format(tpr, tnr))


# # Strategy Based on Labels vs BH

# In[88]:


dfDetail = pd.read_csv("./GOOGL_weekly_return_volatility_detailed.csv")
dfYear2 = dfDetail[dfDetail.Year == 2020]
year2.label = labelYear2

## Add label to detail
labelMap = {}
for (y, w, l) in zip(year2.Year, year2.Week_Number, year2.label):
    key = (y, w)
    value = l
    labelMap[key] = value

temp = []
for (y, w) in zip(dfYear2.Year, dfYear2.Week_Number):
    key = (y, w)
    temp.append(labelMap[key])

## Extract data
dfYear2['Label'] = temp
dfYear2 = dfYear2[['Year', 'Week_Number', 'Close', 'Label']]

## Cut goo2020
goo2020Week = []
for i in range(53):
    temp = dfYear2[dfYear2.Week_Number == i]
    temp = temp.reset_index(drop=True)
    goo2020Week.append(temp)


# In[89]:


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


if __name__ == "__main__":
    # Trading base on my label
    total = proficCalculator(goo2020Week, 100)
    print("Using Label: {}".format(total))

    # Buy and hold 
    first = goo2020Week[0]
    first = first.Close[0]

    last = goo2020Week[-1]
    last = last.Close[len(last)-1]

    r = 1 + (last - first) / last
    total = 100 * r
    print("Buy on first day and Sell on last day: {}".format(total))

