#!/usr/bin/env python
# coding: utf-8

# # Problem for Covid - 19 Data Analysis Project using Python

# # Import the necessary libraries

# In[1]:


#install lib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
url = 'https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv'


# # 1 . Import the dataset using Pandas

# In[2]:


df = pd.read_csv(url)
df


# # 2. High level Data Understanding:

# a. Find no. of rows and columns in the dataset

# In[3]:


print("no. of rows :",len(df))
print("no. of columns :",len(df.columns))


# In[4]:


df.shape


# b. Data types of columns.

# In[5]:


df.dtypes


# c. Info & describe of data in dataframe.

# In[6]:


df.info()


# In[7]:


df.describe(include='all')


# # 3. Low Level Data Understanding:

# a. Find count of unique values in location column.

# In[8]:


df.location.nunique()


# b. Find which continent has maximum frequency using values counts.

# In[9]:


b=df["continent"].dropna()


# In[10]:


a=b.value_counts()


# In[11]:


for i in b:
    if a[i]==a.max():
        print(f"{i} continent has maximum frequency",":",a.max())
        break


# c. Find maximum and mean value in total_case.

# In[12]:


print("Maximum values :",df["total_cases"].max())


# In[13]:


print("Mean values :",df["total_cases"].mean())


# d. Find 25%,50%,and 75% quartile values in total_deaths.

# In[14]:


df["total_deaths"].quantile([0.25,0.5,0.75])


# e. Find which continent has maximum human_development_index.

# In[15]:


a=df[["continent",'human_development_index']].dropna()
a[a['human_development_index'] == a['human_development_index'].max()].drop_duplicates()


# f. Find which continent has minimum gdp_per_capita.

# In[16]:


a=df[["continent",'gdp_per_capita']].dropna()
a[a['gdp_per_capita'] == a['gdp_per_capita'].min()].drop_duplicates()


# # 4.Filter the dataframe with only this columns
# # ['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index'] and update the data frame.

# In[17]:


a= df.filter(['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index'])


# In[18]:


df2=a


# In[19]:


df2


# # 5.Data Cleaning

# a.Remove all duplicates observations.

# In[20]:


df2.drop_duplicates()


# b. Find missing values in all columns.

# In[21]:


df2.isnull()


# c. Remove all observations where continent column value is missing.

# In[22]:


df4=df2.dropna(subset=['continent'])
df4


# d. Fill all  missing values with 0.

# In[23]:


df3=df4.fillna(0)
df3


# # 6. Data time format .

# a. Convert date column in datetime format using pandas.to_datetime.

# In[24]:


import datetime as dt 
df3["date"]=pd.to_datetime(df3["date"]) 
df3["date"]=df3["date"]


# In[25]:


df3


# In[26]:


dates = pd.to_datetime(df3['date'])


# In[27]:


dates


# b. Create new column month after extracting month data from date column.

# In[28]:


df3['month'] = pd.DatetimeIndex(df3['date']).month


# In[29]:


df3


# # 7. Data Aggregation:

# a. Find max value in all columns using groupby function on "continent" column.

# In[30]:


df3.reset_index(inplace = True)
b=df3.groupby(df3['continent']).apply(max)
b


# b. Store the result in a new dataframe named "df_groupby'.

# In[31]:


df_groupby = b
df_groupby


# # 8. Feature Engineering :

# a. Create a new feature "total_deaths_to_total_cases" by ratio of "total_deaths" column to "total_cases".

# In[32]:


df_groupby["total_deaths_to_total_cases"] = df_groupby["total_deaths"]/df_groupby["total_cases"]


# In[33]:


df_groupby


# # 9. Data Visualization :

# a. Perform Univariate analysis on "gdp_per_capita" column by plotting histogram using seaborn dist plot.

# In[34]:


sns.distplot(df_groupby['gdp_per_capita'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram")
plt.show()


# In[35]:


df_groupby.mean()


# b. Plot a scatter plot of "total_cases" & "gdp_per_capita".

# In[36]:


sns.scatterplot(x = 'total_cases', y= "gdp_per_capita", data=df_groupby)
plt.title("Scatter Plot")
plt.xlabel('total_cases')
plt.ylabel('gdp_per_capita')
plt.show()


# c. Plot Pairplot on df_groupby dataset.

# In[37]:


sns.pairplot(df_groupby)


# d. Plot a bar plot of 'continent' column with 'total_cases'.

# In[38]:


g = sns.catplot(x= "continent",y ="total_cases",kind="bar", data=df_groupby)


# # 10. Save the df_groupby dataframe in your local drive using pandas.to_csv function.

# In[39]:


df3.to_csv('df_groupby.csv', index=False)

