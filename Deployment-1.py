#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle as pkl


# In[11]:


data = {
    'Marks': [77,78,79,77,85,45,69,88,74,98,99,90,95,62,41],
    'Age':[21,22,22,23,20,21,22,25,27,27,26,21,25,27,20],
    'Salary': [250,250,220,236,247,280,247,299,288,299,300,300,124,147,188]
}
df = pd.DataFrame(data)


# In[12]:


df


# In[13]:


X=df.drop(columns='Salary',axis=1)
y=df.Salary


# In[14]:


X


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[21]:


print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)


# In[26]:


with open('deploylr.plk','wb') as file:
    pkl.dump(lr,file)

