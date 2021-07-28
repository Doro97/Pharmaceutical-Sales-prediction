#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


# In[90]:


get_ipython().run_line_magic('store', 'train_df')


# # Data

# In[3]:


sample_sub=pd.read_csv('sample_submission.csv')
sample_sub.head()

print(sample_sub.head(), "\n", sample_sub.shape)


# In[4]:


store=pd.read_csv('store.csv')
print(store.head(), "\n", store.shape)


# In[5]:


test=pd.read_csv('test.csv')
print(test.head(), "\n", test.shape)


# In[6]:


train=pd.read_csv('train.csv')
print(train.head(), "\n", train.shape)


# In[7]:


#merge the store dataset with the train dataset

train_df = pd.merge(left=store, right=train, left_on='Store', right_on='Store')
display(train_df)


# In[8]:


#merge the store dataset with the train dataset

test_df = pd.merge(left=store, right=test, left_on='Store', right_on='Store')
test_df


# # Exploratory Data Analysis

# In[9]:


# number of data points
print(f" Number of rows:  {train_df.shape[0]} \n Number of columns: {train_df.shape[1]} ")
#statistical summary of the columns
train_df.describe()


# In[10]:


train_df.info()


# In[11]:


#convert timestamp to datetime format and inserts it the column 'timestamp'
def convert_str_datetime(df):
    df['Date']=pd.to_datetime(df.Date)
    return df
    
convert_str_datetime(train_df)    


# In[12]:


#number of unique values per column
def unique_values(df):
    unique_values=pd.DataFrame(df.apply(lambda x: len(x.value_counts(dropna=False)), axis=0), 
                           columns=['Unique Value Count']).sort_values(by='Unique Value Count', ascending=True)
    return unique_values

unique_values(train_df)


# In[13]:


# duplicated data
def duplicated_data(df):
    return df[df.duplicated()].sum()

duplicated_data(train_df)


# There are no duplicated rows in this dataframe

# In[14]:


# the percentage of missing values in the dataset
def missing_values(x):
    # Total number of elements in the dataset
    totalCells = x.size
    #Number of missing values per column
    missingCount = x.isnull().sum()
    #Total number of missing values
    totalMissing = missingCount.sum()
    # Calculate percentage of missing values
    print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

missing_values(train_df)


# In[15]:


def column_missingdata(df):
    #check for missing values per column
    values=df.isnull().sum().sort_values(ascending=False)
    #percentage of missing values per column
    percentage=df.isnull().mean()*100
    return values,percentage

column_missingdata(train_df)


# ### Dealing with missing values

# In[16]:


#selecting rows where the row CompetitionDistance has no missing values
#train_df=train_df[train_df['CompetitionDistance'].notna()]
#train_df


# 2642 rows have been dropped

# In[17]:


#replace missing values with mean 
def fill_mean(dataframe,column):
    dataframe[column].fillna(dataframe[column].mean(), inplace = True)
    return dataframe

# replacing with the median
def fill_median(dataframe,column):
    dataframe[column].fillna(dataframe[column].median(), inplace = True)
    return dataframe

#fill with mode
def fill_mode(dataframe,column):
    dataframe[column].fillna(dataframe[column].mode(), inplace = True)
    return dataframe

#replaces all missing values with 0
def fill_with_0(dataframe):
    dataframe.fillna(0, inplace=True)
    return dataframe


# In[18]:


train_df=fill_mean(train_df,'CompetitionDistance')
train_df=fill_mean(train_df,'CompetitionOpenSinceMonth')
train_df=fill_mean(train_df,'CompetitionOpenSinceYear')
train_df=fill_mode(train_df,'Promo2SinceWeek')
train_df=fill_median(train_df,'Promo2SinceYear')
train_df=fill_median(train_df,'Promo2SinceWeek')
fill_with_0(train_df)


# In[19]:


column_missingdata(train_df)


# ### Feature creation

# In[1]:


def features_created(data):
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear


# In[20]:


features_created(train_df)


# # Graphical Analysis

# In[21]:


# outliers
def boxplot(df,col):
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.boxplot(y=df[col])


# In[22]:


boxplot(train_df,'Sales')
boxplot(train_df,'CompetitionDistance')
boxplot(train_df,'CompetitionOpenSinceMonth')
boxplot(train_df,'CompetitionOpenSinceYear')
boxplot(train_df,'Promo2SinceWeek')
boxplot(train_df,'Customers')
boxplot(train_df,'Promo2')
boxplot(train_df,'DayOfWeek')
boxplot(train_df,'SchoolHoliday')


# In[23]:


train_df.columns


# In[24]:


#plots
def barplot(x,y,data):
    sns.barplot(x=x,y=y,data=data) 
    plt.show()
    
def count_plot(x,data):
    plt.figure(figsize=(8,5))
    sns.countplot(x=x,data=data)    


# ### Check for distributions in both training and test sets - are the promotions distributed similarly between these two groups?
# 

# In[25]:


# promotions,
fig,ax=plt.subplots(figsize=(10,8))
sns.distplot(test['Promo'],bins=20,color='blue')
ax.set(ylabel='Frequency')
ax.set(xlabel='Promotions')


# In[26]:


fig,ax=plt.subplots(figsize=(10,8))
sns.distplot(train_df['Promo'],bins=20,color='blue')
ax.set(ylabel='Frequency')
ax.set(xlabel='Promotions')


# The distributions of promotions  the train and test set are similar

# ### Check and compare sales behavior before,during and after holidays

# In[27]:


holidays_df=train_df[['Sales','StateHoliday','SchoolHoliday']]
holidays_df.head()
# comparison between sales on holidays and on other days
sns.barplot(x='StateHoliday',y='Sales',data=holidays_df) 
plt.show()
holidays_df['StateHoliday'].unique()   


# In[28]:


# combine the value '0' with the value 0
train_df['StateHoliday'].loc[train_df['StateHoliday']==0]='0'
train_df['StateHoliday'].unique() 


# In[29]:


# sales during other days
other_days=holidays_df.loc[holidays_df['StateHoliday']=='0']
#sales duting holidays
holidays=holidays_df.loc[holidays_df['StateHoliday']!='0']

#plots
sns.barplot(x='StateHoliday',y='Sales',data=holidays) 
plt.title('Sales during the holidays and other days')
plt.show()
    
#groupby during holidays, after holidays(1 week), before holidays(1 week)


# ### Seasonal Purchasing behaviours

# In[30]:


store_1=train_df.loc[(train_df['Store']==1)& (train_df['Sales']>0),['Date','Sales']]
store_10=train_df.loc[(train_df['Store']==10)& (train_df['Sales']>0),['Date','Sales']]
fig=plt.figure(figsize=(18,10))
ax1=fig.add_subplot(211)
ax1.plot(store_1['Date'],store_1['Sales'],'-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Sales')
ax1.set_title('Store 1 sales distribution')

ax2=fig.add_subplot(212)
ax2.plot(store_10['Date'],store_10['Sales'],'-')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
ax2.set_title('Store 10 sales distribution')


# In[31]:


#before christmas
start_date='2014-12-14'
end_date='2014-12-31'
mask=(train_df['Date']>start_date)&(train_df['Date']<=end_date)
beforeChristmas=train_df.loc[mask ]
beforeChristmas

beforeChristmaswk=beforeChristmas.loc[(beforeChristmas['Store']==1)& (beforeChristmas['Sales']>0),['Date','Sales']]
fig=plt.figure(figsize=(18,10))
ax1=fig.add_subplot(211)
ax1.plot(beforeChristmaswk['Date'],beforeChristmaswk['Sales'],'-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Sales')
ax1.set_title('Before Christmas sales distribution')


# In[33]:


#week before christmas
#2014-12-14 to 2014-12-21
#christmas week
#2014-12-22 to 2014-12-28
#week after christmas
#2014-12-31 to 2015-1-5


# In[34]:


# sales and customers during christmas
#groupby christmas holiday and compare sales with other days, seasonality during christmas 


# ### Correlation between sales and number of customers

# In[32]:


#creation of a correlation matrix
corrM=train_df.corr()
corrM


# In[33]:


matrix=np.triu(train_df.corr())

sns.heatmap(corrM,annot=True,square=True,mask=matrix)
fig=plt.gcf()
figsize=fig.get_size_inches()
fig.set_size_inches(figsize*3)
plt.show()


# The number of customers and volume of sales are highly correlated with a score of 89%. This indicates that the more the number of customers the higher the level of sales

# ### How does promo affect sales

# From the correlation matrix above promo and sales are positively correlated with a score of 45%. Therefore an a promotion leads to an increase in sales volume

# In[34]:


sns.barplot(x='Promo',y='Sales',data=train_df) 
plt.title('Comparison between sales during promotions')
plt.show()


# 1 represents a promotion being ran while 0 represents no promotion. From the above graph, the sales volume is increased by about 50% when there is a promotion running.
# This could be an indicator that the promotion is attracting new customers and also serves as a reminder to existing customers to buy more product

# ### Which stores should the promos be deployed in
#     
#     
# 

# In[35]:


count_plot(x='StoreType',data=train_df)
plt.title('Count per store model')


# In[36]:


barplot(x='StoreType',y='Sales',data=train_df) 
plt.show()


# The model 'a' type are the most while model 'b' are the least. However store model 'b' contributes the most to the sales volume

# In[37]:


#stores that ran a promotion
store_promo=train_df.loc[train_df['Promo']==1]
store_promo


# In[38]:


barplot(x='StoreType',y='Sales',data=store_promo) 
plt.show()


# In[39]:


count_plot(x='StoreType',data=store_promo)
plt.title('Count per store model')


# In[45]:


#str_promo.plot(x="StoreType",kind='bar',stacked=True)


# ### Trends of customer behaviour during store open and closing times

# In[40]:


train_df.head()


# In[41]:


#splitting date column into day_name,month,weekdef features_create(data):
def features_create(data): 
    data['WeekOfYear'] = data.Date.dt.weekofyear
    data['year']=train_df['Date'].dt.year
    data['month']=train_df['Date'].dt.month
    data['day_name']=train_df['Date'].dt.day_name()
    return data
features_create(train_df)


# In[42]:



train_df.columns


# # Which stores are opened on all weekdays

# In[43]:


# creates a 'Weekend' column where 0 represents a weekday and 1 a weekend
train_df.loc[(train_df['day_name']=='Saturday')|(train_df['day_name']=='Sunday'),'Weekend']=1
train_df['Weekend']=train_df['Weekend'].fillna(0)
train_df


# In[44]:


sns.barplot(y='Sales',x='StoreType',hue='Weekend',data=train_df)
plt.title('Comparison of sales per store on weekends and weekdays')


# In[45]:


sns.barplot(y='Sales',x='day_name',hue='StoreType',data=train_df)
plt.xticks(rotation=90)


# In[46]:



#dataframe where a store is closed on any day(open=0)
closedstore_df=train_df.loc[(train_df['Open']==0)].copy()
closedstore_df['StoreType'].unique()
closedstore_df

sns.countplot(x='StoreType',data=closedstore_df)
plt.xticks(rotation=90)


# In[47]:


#dataframe where a store is closed on any weekday(open=0,weekend=0)
closedstoreWeekend_df=train_df.loc[(train_df['Open']==0) & (train_df['Weekend']==0)].copy()
closedstoreWeekend_df['StoreType'].unique()
closedstoreWeekend_df

sns.countplot(x='StoreType',data=closedstoreWeekend_df)
plt.xticks(rotation=90)


# The model type 'a' has the most stores closed the most during the weekdays while model type 'b' has the least number of store closed during the week.This could explain why model type 'b' has the largest sales  volume

# #### How does this affect their sales on weekends

# In[48]:


#plot of sales in closed stores during the weekend
closedstoreWeekend_df=train_df.loc[(train_df['Open']==1) & (train_df['Weekend']==1)].copy()
closedstoreWeekend_df['StoreType'].unique()
closedstoreWeekend_df

sns.barplot(y='Sales', x='StoreType',data=closedstoreWeekend_df)
plt.xticks(rotation=90)


# From the above plots, model type 'b' has the most sales volume during the weekend, while model type 'a ' has the least sales volume during the weekend

# ### Check how assortment type affects sales

# In[49]:


sns.barplot(y='Sales',x='Assortment',data=train_df)
plt.xticks(rotation=90)


# The assortment levels have been categorised as follows a=basic, b=extra, c= extended. The extra assortment type has the most sales while the basic assortment type has the least sales volume

# ### How does the distance to the next competitor affect sales

# In[50]:


train_df


# In[51]:


#relationship betweeen cdistance from competitor and sales
competitorDist=train_df[['CompetitionDistance','Sales']].copy()
#correlation matrix
#creation of a correlation matrix
corr=competitorDist.corr()
corr


# The correlation between CompetitionDistance and Sales is -1.9% which indicates that there is very little association between the distance from competition and the level of sales. The negative indicates that there is a weakly negative correlation therefore if the distance from competition is large then the  level of sales is will reduce

# #### What if the store and its competitors are in the same city? does distance matter?
# 

# In[52]:


#assuming the distance between a store to its competition is less than 10000meters
competition=competitorDist.loc[(competitorDist['CompetitionDistance']<=10000)]
towncorr=competition.corr()
towncorr


# In[53]:


sns.heatmap(towncorr,annot=True,square=True)
fig=plt.gcf()
figsize=fig.get_size_inches()
plt.show()


# The distance does not matter if the store and its competitors are in the same city since there is still very little correlation between distance and sales

# ### How does the opening and reopening of new competitors affect stores

# In[60]:


store.head()


# In[55]:


store_nocompetition=store.loc[store['Store']==879]
store_nocompetition


# In[56]:


s=train_df.loc[(train_df['Store']==291)& (train_df['CompetitionDistance']!='NaN'),['Store','Sales','CompetitionDistance']]
s['CompetitionDistance'].unique()


# In[ ]:




