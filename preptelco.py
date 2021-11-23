#!/usr/bin/env python
# coding: utf-8

# # Prepare Data: 
# - acquire data 
# - preapare data 
#     1. summarize data 
#         - head(), describe(), info(), isnull(), value_count(), shape
#         - plt.hist(), plt.boxplot, pandasgui
#         - document takeways 
#     
#     2.  clean 
#         - missing values
#         - outliers
#         - data error
#         - tidy data 
#         - create new variables 
#         - rename columns 
#         - scale numeric columns 
#             - SPLIT 
#                 - train
#                 - test
#                 - validate 
# 
#         

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquiretelco


# In[2]:


df = acquiretelco.get_telco_data()


# In[3]:


df.shape


# In[4]:


# Get information about the dataframe: column names, rows, datatypes, non-missing values.
df.info()


# In[5]:


# Get summary statistics for numeric columns.

df.describe()


# In[6]:


# df.T


# In[7]:


# Check out distributions of numeric columns.

num_cols = df.columns[[df[col].dtype == 'int64' for col in df.columns]]
for col in num_cols:
    plt.hist(df[col])
    plt.title(col)
    plt.show()


# In[21]:


# Use .describe with object columns.

obj_cols = df.columns[[df[col].dtype == 'O' for col in df.columns]]
for col in obj_cols:
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True, dropna=False))
    print('----------------------')


# In[22]:


# Find columns with missing values and the total of missing values

missing = df.isnull().sum()
missing[missing > 0]


# In[23]:


# Drop duplicates and check the shape of  data.

df = df.drop_duplicates()
df.shape


# In[11]:


df.head()


# In[15]:


# Get information about the dataframe: column names, rows, datatypes, non-missing values.
df.info()


# In[71]:


# Split 
def telco_split(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test df
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test


# In[76]:


# Prepare data 
def telco_prep(df): 
    '''
    This function reads a dataframe of telco data and
    returns prepped train, validate, and test
    '''
    # Drop null values stored as whitespace          
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Change dtype from non-numeric to numeric using .replace 
    df = df.replace({'gender': {'Male':1, 'Female':0}})
    df = df.replace({'partner':{'Yes':1, 'No': 0}})
    df = df.replace({'dependents':{'Yes':1, 'No': 0}})
    df = df.replace({'phone_service': {'Yes':1, 'No':0}})
    df = df.replace({'tech_support':{'Yes':1, 'No':0, 'No internet service':0} })
    df = df.replace({'churn':{'Yes':1, 'No': 0}})
    df = df.replace({'streaming_tv': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'multiple_lines': {'Yes':1, 'No':0, 'No phone service':0}})
    df = df.replace({'online_security': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'online_backup': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'device_protection': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'streaming_movies': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'paperless_billing': {'Yes':1, 'No':0, 'No internet service':0}})
    df = df.replace({'payment_type': {'Mailed check': 0, 'Electronic check':0, 'Bank transfer (automatic)':1, 'Credit card (automatic)':1}})
    
    dummies_internet_type = pd.get_dummies(df.internet_service_type)
    df = pd.concat([df, dummies_contract_type, dummies_internet_type], axis = 1)
    
    drop_columns = ['internet_service_type', 'contract_type', 'payment_type_id', 
                    'internet_service_type_id','contract_type_id','gender']
    df = df.drop(columns = cols_to_drop)
    
    
    train, test, validate = split_data(df)
    return train, test, validate


# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




