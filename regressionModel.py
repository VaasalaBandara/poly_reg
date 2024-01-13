#!/usr/bin/env python
# coding: utf-8

# the regression models require a data set that consists of one independent variable and dependent variable

# In[1]:


import pandas as pd


# In[2]:


data={'income_owners':[60,85.5,64.8,61.5,87,110.1,108,82.8,68,93,51,81],'lotsize_owners':[18.4,16.8,21.6,20.8,23.6,19.2,17.6,22.4,20,20.8,22,20],'income_nonowners':[75,52.8,64.8,43.2,84,49.2,59.4,66,47.4,33,51,63],'lotsize_nonowners':[19.6,20.8,17.2,20.4,17.6,17.6,16,18.4,16.4,18.8,14,14.8]}


# In[3]:


print(data)


# In[4]:


df=pd.DataFrame(data)


# In[5]:


print(df)


# the dataset has two categories owners vs non owners and each category has 2 independent variables. to shrink the dataset into two variables we consider the difference

# In[6]:


df_owners=df['income_owners']-df['lotsize_owners']


# In[7]:


print(df_owners)


# In[8]:


df_nonowners=df['income_nonowners']-df['lotsize_nonowners']


# In[9]:


print(df_nonowners)


# In[12]:


data2={'data_owners':df_owners,'data_nonowners':df_nonowners}


# In[13]:


df2=pd.DataFrame(data2)


# In[17]:


print(df2)


# choosing the proper regression model

# In[15]:


import matplotlib.pyplot as plt


# In[23]:


plt.scatter(df_owners,df_nonowners,s=10)
plt.show()


# In[32]:


X=np.array(df_owners)
X=X.reshape(-1,1)
print(X)


# In[33]:


y=np.array(df_nonowners)
print(y)


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


lin_reg1=LinearRegression()


# In[36]:


lin_reg1.fit(X,y)


# In[37]:


print(lin_reg1.score(X,y))


# the low score indicates that the linear regression model is not suitsable for this dataset

# checking the polynomial regression model

# In[51]:


from sklearn.preprocessing import PolynomialFeatures
polyregr2=PolynomialFeatures(degree=2)
polyregr3=PolynomialFeatures(degree=3)
polyregr4=PolynomialFeatures(degree=4)
polyregr5=PolynomialFeatures(degree=5)
polyregr7=PolynomialFeatures(degree=6)


# In[52]:


poly2=polyregr2.fit_transform(X)
poly3=polyregr3.fit_transform(X)
poly4=polyregr4.fit_transform(X)
poly5=polyregr5.fit_transform(X)
poly7=polyregr7.fit_transform(X)


# In[53]:


#second degree polynomial regression
lin_reg2=LinearRegression()
lin_reg2.fit(poly2,y)
print(lin_reg2.score(poly2,y))


# In[54]:


#third degree polynomial regression
lin_reg3=LinearRegression()
lin_reg3.fit(poly3,y)
print(lin_reg3.score(poly3,y))


# In[55]:


#fouth degree polynomial
lin_reg4=LinearRegression()
lin_reg4.fit(poly4,y)
print(lin_reg4.score(poly4,y))


# In[56]:


#fifth degree polynomial
lin_reg5=LinearRegression()
lin_reg5.fit(poly5,y)
print(lin_reg5.score(poly5,y))


# The score appears to be increasing with the higher degree of the polynomial

# In[57]:


#seventh degree polynomial
lin_reg7=LinearRegression()
lin_reg7.fit(poly7,y)
print(lin_reg7.score(poly7,y))


# the polynomial regression model appears be unsuitable for the dataset since the score is less than 0.5

# In[ ]:




