#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()
x = iris.data
y = iris.target
x.shape


# In[3]:


type(x)


# In[4]:


type(y)


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.shape


# In[6]:


x_test.shape


# In[7]:


x_test


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:





# In[9]:


y_predict = model.predict(x_test)
y_predict


# In[10]:


model.score(x_test,y_test)


# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm


# In[12]:


plt.figure(figsize=(8,4))
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.xlabel('Predict')
plt.ylabel('Actual')


# In[13]:


#As we can see our model get 1.00 Score means none prediction of every row is incorrect.
#This heatmap shows all predicted values matched with the actual values


# In[18]:


from sklearn.metrics import precision_score
precision = precision_score(y_test,y_predict,average='macro')
precision


# In[21]:


from sklearn.metrics import recall_score
recall = recall_score(y_test,y_predict,average='macro')
recall


# In[22]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test,y_predict,average='macro')
f1

