#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np


# In[28]:


dfLabel = pd.read_csv("./GOOGL_weekly_return_volatility.csv")
dfDetail = pd.read_csv("./GOOGL_weekly_return_volatility_detailed.csv")


# In[31]:


# Add label to detail
labelMap = {}
for (y, w, l) in zip(dfLabel.Year, dfLabel.Week_Number, dfLabel.label):
    key = (y, w)
    value = l
    labelMap[key] = value

temp = []
for (y, w) in zip(dfDetail.Year, dfDetail.Week_Number):
    key = (y, w)
    temp.append(labelMap[key])

# Extract data
dfDetail['Label'] = temp
dfDetail = dfDetail[['Year', 'Week_Number', 'Close', 'Label']]
dfDetail


# In[32]:


# Cut into two year
goo2019 = dfDetail[:252]
goo2020 = dfDetail[252:]
goo2020 = goo2020.reset_index(drop=True)

# Cut into week 
## Cut goo2019
goo2019Week = []
for i in range(53):
    temp = goo2019[goo2019.Week_Number == i]
    temp = temp.reset_index(drop=True)
    goo2019Week.append(temp)
## Cut goo2020
goo2020Week = []
for i in range(53):
    temp = goo2020[goo2020.Week_Number == i]
    temp = temp.reset_index(drop=True)
    goo2020Week.append(temp)

## Combine two year
gooWeek = goo2019Week + goo2020Week
gooWeek[:10]


# In[33]:


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

