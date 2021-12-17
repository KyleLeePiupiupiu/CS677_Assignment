#!/usr/bin/env python
# coding: utf-8

# # Open the file
# My BU ID is U64501194
# Therefore, I am dealing with the silver medals

# ### 1. load the "country medals" csv as a list of lines using Python and construct a sublist for you group

# In[35]:


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os

ticker='Country_Medals'
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


# In[36]:


lines[0:5]


# # Extract the data 

# In[37]:


# Class a data type (year, countryName, silverMetal)
class DataNode:
    def __init__(self, year, countryName, silver):
        self.year = year
        self.countryName = countryName
        self.silver = silver
    def show(self):
        print("{:5}  {:15}  {:3}".format(self.year, self.countryName, self.silver))


# In[38]:


# Split the raw data into list
table = []
for line in lines:
    temp = line.split(',')
    table.append(temp)
  
# Extract the raw data
extractData = []
for row in table:
    y = row[0]
    cn = row[2]
    s = row[6]
    dataNode = DataNode(y, cn, s)
    extractData.append(dataNode)

# Remove the title from the extractData
extractData = extractData[1:]


# In[39]:


# Show the first 20 row of the extracted data
for i in range(20):
    extractData[i].show()


# # Analysis 

# ### 2. how many entries are there?

# In[40]:


# How many entries
entries = len(extractData)
print("we have {} entries of data and 1 row of title".format(entries))


# Calculate the average medals for each country

# In[41]:


# Sort the list with the key 'countryName'
sortData = sorted(extractData, key= lambda x:x.countryName)
# Show first 20 data
for node in sortData[0:20]:
    node.show()


# In[42]:


# Extract the country from data
temp = []
for node in sortData:
    c = node.countryName
    temp.append(c)


# Processing the data into the form (countryname, list(a series of medal wins)
countrySet = sorted(set(temp))
medalList = [[] for i in range(len(countrySet))]

for node in sortData:
    n = node.countryName
    s = node.silver
    index = countrySet.index(n)
    medalList[index].append(int(s))
    
countryMedalData = list(zip(countrySet, medalList))


# In[43]:


# Print the first 20 countryMedalData
for data in countryMedalData[0:20]:
    print(data)


# ### 3. compute the average numbers of medals per country and write this (in decreasing order) to a le "average medals per country.csv" for your group
# 
# ### 4. which country has the highest (average) number of medals?
# ### 5. list top 10 countries by (averaged) number of medals

# In[44]:


# Average medal for each country
averageMedal = []
for data in countryMedalData:
    name = data[0]
    medal = data[1]
    aveMedal = sum(data[1]) // len(data[1])
    averageMedal.append((name, aveMedal))
    


# In[45]:


# Sorting the averageMedal in decreasing ordert
temp = sorted(averageMedal, key=lambda x:x[1], reverse = True)
print(temp)


# In[46]:


print("\n\n\n\n{} has won the most averaged silver medal".format(temp[0][0]), )
print("Unified Team only once participated the olympic game in 1992, and the team won 38 silver metals")


# In[47]:


# Wirte the average data into a new cvs name ""average medals per country.csv"
import csv
try:
    with open('average_medals_per_country.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Country Name', 'Average Silver Medal'])
        for data in temp:
            name = data[0]
            medal = data[1]
            writer.writerow([name, medal])
    print('\nOutput csv SUCCESSFUL\n')
except:
    print("Something wrong when output the csv file")


# In[48]:


# List top 10 countries by (average) number of medals
for data in temp[0:10]:
    print("{:25} {}".format(data[0], data[1]))


# ### 6. compute the median number of medals per country and write this (in decreasing order) to a le "median medals per country.csv" for your group

# In[49]:


def medianNum(inputList):
    inputList = sorted(inputList)
    l = len(inputList)
    if l % 2 == 0:
        index = l // 2
        return (inputList[index] + inputList[index - 1]) // 2
        
    else:
        index = l // 2
        return inputList[index]
        
    
    


# In[50]:


medMedal = []
for data in countryMedalData:
    name = data[0]
    medal = data[1]
    l = len(medal)
    median = medianNum(medal)
    medMedal.append((name, median))

medMedal = sorted(medMedal, key=lambda d:d[1], reverse=True)
print(medMedal)


# In[51]:


# Output csv file
try:
    with open('median_medals_per_country.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Country Name', 'median of Silver Medal'])
        for data in medMedal:
            name = data[0]
            m = data[1]
            writer.writerow([name, m])
    print('\nOutput csv SUCCESSFUL\n')
except:
    print("Something wrong when output the csv file")


# ### 7. which country has the highest median number of medals?

# In[52]:


print("{} has the highest median number of silver medals, which is {}".format(medMedal[0][0], medMedal[0][1]))


# ### 8. list top 10 countries by median number of medals

# In[53]:


for data in medMedal[0:10]:
    country = data[0]
    m = data[1]
    print("{:25}  {:5}".format(country, m))

