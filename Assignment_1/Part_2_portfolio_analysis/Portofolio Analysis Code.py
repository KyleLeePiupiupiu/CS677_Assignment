#!/usr/bin/env python
# coding: utf-8

# # Download the Google stock data

# In[30]:


# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:02:02 2021

@author: epinsky
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""

# install yfinance version 0.1.62
#   !pip install yfinance==0.1.62
# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
import pandas as pd
import yfinance as yf

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
#       yf.pdr_override()
        df = yf.download(ticker, start=start_date, end=end_date)
# can use this as well        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.day_name()
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        num_lines = len(df)
        df = df[col_list]
        print('read ', num_lines, ' lines of data for ticker: ' , ticker)
        return df
    except Exception as error:
        print(error)
        return None

try:
    ticker="GOOGL"
    input_dir = os.getcwd()
    output_file = os.path.join(input_dir, ticker + '.csv')
    df = get_stock(ticker, start_date='2016-01-01', end_date='2020-12-31', 
               s_window=14, l_window=50)
    df.to_csv(output_file, index=False)
    print('wrote ' + str(len(df)) + ' lines to file: ' + output_file)
except Exception as e:
    print(e)
    print('failed to get Yahoo stock data for ticker: ', ticker)


# # Open "SPY.csv" file

# In[31]:


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os

ticker='SPY'
input_dir = os.getcwd()
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
    


# In[32]:


# Show the first 5 rows of the table
spyTable = lines
for i in range(5):
    print(spyTable[i])


# In[33]:


# How many data do we have
spyDataLen = len(spyTable) - 1
print('we have {} data'.format(spyDataLen))


# In[34]:


# Type structure of the table
print("The spyTable is a {} of {}".format(type(spyTable), type(spyTable[0])))


# In[35]:


# Split the row 1 data, which are labels
# Turn a string into a list of string
lables = spyTable[0]
lables = lables.split(",")
print(lables)

# Know the index of label 'Return'
# Know the index of label 'year'
returnIndex = lables.index("Return")
yearIndex = lables.index('Year')
print("\nThe index of label 'Return' is {}".format(returnIndex))
print("\nThe index of label 'Year' is {}".format(yearIndex))


# In[36]:


# Collect all 'Return' data from the spyTable
spyReturnList = []
for line in spyTable[1:]:
    line = line.split(',')
    returnn = line[returnIndex]
    spyReturnList.append(float(returnn))
    
# Check the amount of the return data is correct
# Double check the second day data is the same with the csv file
print(len(spyReturnList))
print(spyReturnList[1])


# # Open "GOOGL.csv" file

# In[37]:


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os

ticker='GOOGL'
input_dir = os.getcwd()
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
    


# In[38]:


# Show the first 5 rows of the table
gooTable = lines
for i in range(5):
    print(gooTable[i])


# In[39]:


# how many data do we have
gooDataLen = len(gooTable) - 1
print('we have also {} data, match with "SPY" data'.format(gooDataLen))


# In[40]:


# Collect all 'Return' data from the spyTable
gooReturnList = []
for line in gooTable[1:]:
    line = line.split(',')
    returnn = line[returnIndex]
    gooReturnList.append(float(returnn))

# Check the amount of the return data is correct
# Double check the second day data is the same with the csv file
print(len(gooReturnList))
print(gooReturnList[1])


# # Portofolio Analysis
# Now we have extracted both 'Return' data from both csv file
# Then, I am going to build the portofolio

# In[41]:


# Locate the index of each year
# First, extract the 'Year' data from raw data, and turn them into a list
yearList = []
yearIndexList = []
for line in spyTable[1:]:
    line = line.split(',')
    year = line[yearIndex]
    yearList.append(int(year))

# Determine the index of each year in the list
for i in range(5):
    yearIndexList.append(yearList.index(2016 + i))
yearIndexList.append(len(spyReturnList))

# Show the list
print(yearIndexList)


#  Define some useful function for convience

# In[42]:


class node:
    def __init__(self, final, mx, mn):
        self.final = final
        self.max = mx
        self.min = mn
        
def finMaxMin(yearlyFund):
    fin = yearlyFund[-1]
    mx = max(yearlyFund)
    mn = min(yearlyFund)
    
    return node(fin, mx, mn)

def comReturnFun(alpha, beta, security1, security2):
    comReturn = []
    for i in range(len(spyReturnList)):
        comReturn.append(alpha * security1[i] + beta * security2[i])
    return comReturn


def fundEDayFun(initialFund, comReturn, yearIndexList):
    fund = [[] for i in range(5)]
    temp = initialFund
    for i in range(5):
        for r in comReturn[yearIndexList[i] : yearIndexList[i + 1]]:
            temp = temp * (1 + r)
            fund[i].append(temp)
    return fund


# In[43]:


# Parameter from the question
initialFund = 100
alphaList = [0,0.2,0.4,0.6,0.8,1] 


# In[44]:


# Process the data
total = []
for alpha in alphaList:
    beta = 1 - alpha
    comReturn = comReturnFun(alpha, beta, spyReturnList, gooReturnList)
    fundFiveYear = fundEDayFun(initialFund, comReturn, yearIndexList)
    result = []
    for i in range(5):
        temp = finMaxMin(fundFiveYear[i])
        result.append(temp)
    total.append(result)
    


# In[45]:


# Print the data in the way we want
for i in range(5):
    print("{}".format(2016+i))
    for j in range(len(alphaList)):
        temp = total[j][i]
        print('Alpha= {:<3}  Final= {:6.2f}  Max= {:6.2f}  Min= {:6.2f}'.format(alphaList[j], temp.final, temp.max, temp.min))
    print()

