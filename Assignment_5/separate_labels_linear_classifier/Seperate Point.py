#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[46]:


df = pd.read_csv("./GOOGL_weekly_return_volatility.csv")
year1 = df[df.Year == 2019]
year2 = df[df.Year == 2020]

# Plot year 1
year11 = year1[year1.label == 1]
year10 = year1[year1.label == 0]


f = plt.figure()
f.set_size_inches(12,24)

y1 = f.add_subplot(2,1,1)
y1.axhline(y=0, color='k')
y1.axvline(x=0, color='k')
y1.set_title('Year1', fontsize=20)
y1.set_xlabel('Mean', fontsize=20)
y1.set_ylabel('Volatility', fontsize=20)

## plot green
marker = list(year11.index)
x = list(year11.mean_return)
y = list(year11.volatility)

plt.scatter(x, y, c="green", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)

## plot red
marker = list(year10.index)
x = list(year10.mean_return)
y = list(year10.volatility)

plt.scatter(x, y, c="red", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)


# In[47]:


# remove 12,43, 9,51,50,47
dropIndex = [12,43,9,51,50,47]
reducedYear1 = year1.drop(labels=dropIndex, axis=0)

# Plot year 1
year11 = reducedYear1[reducedYear1.label == 1]
year10 = reducedYear1[reducedYear1.label == 0]


f = plt.figure()
f.set_size_inches(12,24)

y1 = f.add_subplot(2,1,1)
y1.axhline(y=0, color='k')
y1.axvline(x=0, color='k')
y1.set_title('reducedYear1', fontsize=20)
y1.set_xlabel('Mean', fontsize=20)
y1.set_ylabel('Volatility', fontsize=20)

## plot green
marker = list(year11.index)
x = list(year11.mean_return)
y = list(year11.volatility)

plt.scatter(x, y, c="green", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)

## plot red
marker = list(year10.index)
x = list(year10.mean_return)
y = list(year10.volatility)

plt.scatter(x, y, c="red", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)



## plot a simple classifier line
a = 10
b = 1
x = [-0.3,0,1,]
y = []
for inte in x:
    y.append(inte*a + b)
plt.plot(x, y)
print("funciton is y = {}x + {}".format(a, b))


# In[48]:


# plot year 2 with y = 10x + 1

# Plot year 2
year21 = year2[year2.label == 1]
year20 = year2[year2.label == 0]


f = plt.figure()
f.set_size_inches(12,24)

y1 = f.add_subplot(2,1,1)
y1.axhline(y=0, color='k')
y1.axvline(x=0, color='k')
y1.set_title('Year2', fontsize=20)
y1.set_xlabel('Mean', fontsize=20)
y1.set_ylabel('Volatility', fontsize=20)

## plot green
marker = list(year21.index)
x = list(year21.mean_return)
y = list(year21.volatility)

plt.scatter(x, y, c="green", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)

## plot red
marker = list(year20.index)
x = list(year20.mean_return)
y = list(year20.volatility)

plt.scatter(x, y, c="red", marker='o', s=70)

for i in range(len(x)):
    plt.text(x[i], y[i], marker[i], fontsize=12)

## plot the classifier
a = 10
b = 1
x = [-0.3,0,1,]
y = []
for inte in x:
    y.append(inte*a + b)
plt.plot(x, y)


# # Use simple classifier to trade

# In[49]:


# label prediction
x = year2.mean_return
y = year2.volatility
labelTrue = year2.label
labelPred = []

for (a, b) in zip(x, y):
    temp = 10*a - b + 1
    if temp >= 0:
        labelPred.append(1)
    else:
        labelPred.append(0)


# In[50]:


year2.loc[:, ['label']]


# In[54]:


dfDetail = pd.read_csv("./GOOGL_weekly_return_volatility_detailed.csv")
dfYear2 = dfDetail[dfDetail.Year == 2020]

year2.loc[:, ['label']] = labelPred

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


# In[61]:


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


# In[ ]:




