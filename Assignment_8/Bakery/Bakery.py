#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('./BreadBasket_DMS.csv')
dfoutput = pd.read_csv("./BreadBasket_DMS_output.csv")


# In[4]:


dfoutput


# # What is the busiest day in terms of transactions

# In[5]:


temp = dfoutput.groupby(['Weekday']).count()
temp = temp[['Transaction']]
temp.sort_values(by=['Transaction'])


# In[6]:


temp = dfoutput.groupby(['Hour']).count()
temp = temp[['Transaction']]
temp.sort_values(by=['Transaction'])


# In[7]:


temp = dfoutput.groupby(['Period']).count()
temp = temp[['Transaction']]
temp.sort_values(by=['Transaction'])


# # What is the most profitable time

# In[8]:


temp = dfoutput.groupby(['Weekday']).sum()
temp = temp[['Item_Price']]
temp.sort_values(by=['Item_Price'])


# In[9]:


temp = dfoutput.groupby(['Hour']).sum()
temp = temp[['Item_Price']]
temp.sort_values(by=['Item_Price'])


# In[10]:


temp = dfoutput.groupby(['Period']).sum()
temp = temp[['Item_Price']]
temp.sort_values(by=['Item_Price'])


# # The most and least popular item

# In[11]:


temp = dfoutput.groupby(['Item']).count()
temp.nlargest(5, ['Transaction'])


# In[12]:


temp = dfoutput.groupby(['Item']).count()
temp.nsmallest(10, ['Transaction'])


# # How many barrista
# 
# There are roughly 6*4+1 = 25 weeks in total in this data

# In[13]:


temp = dfoutput.loc[:, ['Weekday', 'Transaction']]
a = temp.groupby(['Weekday'])
weekday = [wd for wd in a]

print('Average customers of each weekday of the week')
for i in range(len(weekday)):
    day = weekday[i][0]
    data = weekday[i][1]
    data = data['Transaction'].tolist()
    data = set(data)
    data = len(data)
    data = data / 25
    print(day, data)


# # Divide in drinks, food, unknown

# In[14]:


temp = dfoutput
temp = temp.Item
temp = list(temp)
temp = set(temp)
print(temp)
print()
# Make category
food = ['Salad','Fudge','Crepes','Bare Popcorn','Chicken sand','Tacos/Fajita','Bread Pudding','Granola','Focaccia','Tiffin','Baguette','Panatone', 'Spanish Brunch','Lemon and coconut', 'Kids biscuit', 'Pick and Mix Bowls','Chocolates', 'Caramel bites','Frittata', 'My-5 Fruit Shoot', 'Cake','Honey', 'Cherry me Dried fruit', 'Raspberry shortbread sandwich','Mighty Protein','Eggs','Truffles', 'Brownie','Scone', 'Bread','Dulce de Leche', 'Sandwich','Empanadas','Scandinavian','Olum & polenta', 'Vegan Feast','Medialuna', 'Vegan mincepie','Cookies','Victorian Sponge','Duck egg', 'Tartine','Crisps',  'Chicken Stew', 'Bacon']
drinks = ['Tea','Muesli','Soup','Juice', 'Coffee','Smoothies','Hot chocolate', 'Mineral water',"Ella's Kitchen Pouches", 'Coke','Extra Salami or Feta','Gingerbread syrup']
unknown = []
for item in temp:
    if (item not in food) and (item not in drinks):
        unknown.append(item)

print('food list')
print(food)
print()

print('drinks list')
print(drinks)
print()

print('unknown list')
print(unknown)
print()


# In[15]:


temp = dfoutput.loc[:, ['Item', 'Item_Price']]
temp = temp.values.tolist()

# food price
price = []
for itemPrice in temp:
    if itemPrice[0] in food:
        price.append(itemPrice[1])
price = sum(price) / len(price)
print(F'food average price is {price}')

# drinks price
price = []
for itemPrice in temp:
    if itemPrice[0] in drinks:
        price.append(itemPrice[1])
price = sum(price) / len(price)
print(F'drinks average price is {price}')



# # Does this coffee shop make more money from selling drinks or food?

# In[16]:


temp = dfoutput.loc[:, ['Item', 'Item_Price']]
temp = temp.values.tolist()


# food price
price = []
for itemPrice in temp:
    if itemPrice[0] in food:
        price.append(itemPrice[1])

price = sum(price) 
print(F'food total profit is {price}')

# drinks price
price = []
for itemPrice in temp:
    if itemPrice[0] in drinks:
        price.append(itemPrice[1])
price = sum(price) 
print(F'drinks total profit is {price}')


# # what are the top 5 most popular items for each day

# In[18]:


temp = dfoutput.loc[:, ['Weekday', 'Item']]
a = temp.groupby(['Weekday'])
weekday = [wd for wd in a]

for i in range(len(weekday)):
    day = weekday[i][0]
    a = weekday[i][1].groupby(['Item']).count()
    a = a.nlargest(5, ['Weekday'])
    print(day)
    print(a)
    print()


# # 5 least popular thing
# 

# In[23]:


temp = dfoutput.loc[:, ['Weekday', 'Item']]
a = temp.groupby(['Weekday'])
weekday = [wd for wd in a]

for i in range(len(weekday)):
    day = weekday[i][0]
    a = weekday[i][1].groupby(['Item']).count()
    a = a.nsmallest(20, ['Weekday'])
    print(day)
    print(a)
    print()


# # How many drinks are there per transaction
# There are 9684 transaction in total

# In[189]:


drinkAmount = np.zeros(9684)
for (transactionNum, item) in zip(dfoutput.Transaction, df.Item):
    if item in drinks:
        drinkAmount[transactionNum-1] += 1
print('avarage drinks per transaction is {}'.format(sum(drinkAmount) / len(drinkAmount)))

