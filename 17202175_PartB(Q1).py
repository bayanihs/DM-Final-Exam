#!/usr/bin/env python
# coding: utf-8

# PART B 

# QUESTION 1

# In[16]:


#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


#load data
data = pd.read_csv('C:/Users/LV/Desktop/Data(Exam).csv')


# correlation method

# In[17]:


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(17,17))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[18]:


#Correlation with output variable
cor_target = abs(corrmat ["Type"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.6]
relevant_features


# chi square method

# In[19]:


X = data.iloc[:,1:17]  #independent columns
y = data.iloc[:,17]    #target column:type


# In[20]:


#apply SelectKBest class to extract top 4 best features
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(4,'Score'))  #print 4 best features


# anova method

# In[21]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[22]:


# Select Features With Best ANOVA F-Values

# Create an SelectKBest object to select features with 4 best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=4)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X, y)


# In[23]:


print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])


# In[33]:


print(X_kbest[0:5,:])


# conclusion for Question 1: both correlation and Chi-square method gave different output

# QUESTION 2

# In[43]:


from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Using result in Correlation, to test on NB classification algorithm

# In[73]:


df=data[['Eggs','Milk','Backbone','Tail','Type']] #select the relavent features & target only
features= ['Eggs','Milk','Backbone','Tail']
target = 'Type'


# In[74]:


features_train, features_test, target_train, target_test = train_test_split(df[features],
df[target],test_size = 0.33,random_state = 54)


# In[75]:


model = GaussianNB()
model.fit(features_train, target_train)
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)


# Using result in Chisquare, to test on NB classification algorithm

# In[76]:


df2=data[['Legs','Feathers','Fins','Milk','Type']] #select the relavent features & target only
features2= ['Legs','Feathers','Fins','Milk']
target2 = 'Type'


# In[77]:


features_train, features_test, target_train, target_test = train_test_split(df2[features2],
df2[target2],test_size = 0.33,random_state = 54)


# In[78]:


model = GaussianNB()
model.fit(features_train, target_train)
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)


# Based on accuracy result tested, Chi-Square method has the higher accuracy.
# Hence it can be conclude that the select features are Legs, Feathers, Fins, Milk
