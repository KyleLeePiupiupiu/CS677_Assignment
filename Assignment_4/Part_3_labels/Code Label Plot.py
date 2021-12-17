#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("./GOOGL_weekly_return_volatility.csv")
dfYear1 = data[:53]
dfYear2 = data[53:]
dfYear2 = dfYear2.reset_index(drop=True)


# In[3]:


# Plot year 1
green = dfYear1[dfYear1.label == 1]
red = dfYear1[dfYear2.label == 0]

# Set plot size
f = plt.figure()
f.set_size_inches(12,24)

## Green
### Initialize
y1 = f.add_subplot(2,1,1)
y1.axhline(y=0, color='k')
y1.axvline(x=0, color='k')
y1.set_title('Year1', fontsize=20)
y1.set_xlabel('Mean', fontsize=20)
y1.set_ylabel('Volatility', fontsize=20)
### plot green
x = green.mean_return
y = green.volatility
text = green.Week_Number
y1.scatter(x=x, y=y, c="green", marker='o', s=70)
### plot red
x = red.mean_return
y = red.volatility
text = red.Week_Number
y1.scatter(x=x, y=y, c="red", marker='o', s=70)


# In[4]:


# Plot Year 2
green = dfYear2[dfYear1.label == 1]
red = dfYear2[dfYear2.label == 0]

# Set plot size
f = plt.figure()
f.set_size_inches(12,24)

### Initialize
y1 = f.add_subplot(2,1,1)
y1.axhline(y=0, color='k')
y1.axvline(x=0, color='k')
y1.set_title('Year2', fontsize=20)
y1.set_xlabel('Mean', fontsize=20)
y1.set_ylabel('Volatility', fontsize=20)
### plot green
x = green.mean_return
y = green.volatility
text = green.Week_Number
y1.scatter(x=x, y=y, c="green", marker='o', s=70)
### plot red
x = red.mean_return
y = red.volatility
text = red.Week_Number
y1.scatter(x=x, y=y, c="red", marker='o', s=70)


