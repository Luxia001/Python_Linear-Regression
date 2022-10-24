#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Sales_masked.csv')


# In[3]:


df.head()


# In[4]:


df.DayKey=pd.to_datetime(df.DayKey,format='%d/%m/%Y')


# In[5]:


df.set_index('DayKey',inplace=True)


# In[6]:


y10m1 = pd.read_csv('Month-2010/2010-01.csv')


# In[7]:


y10m1.dropna(inplace=True)
y10m1.DayKey=pd.to_datetime(y10m1.DayKey,format='%d/%m/%Y')


# In[8]:


y10m1.set_index('DayKey',inplace=True)


# In[9]:


df


# In[10]:


df[(df['ProductGroup']=='G6') & (df['Area']=='A1')& (df['Channel']=='C1')].loc['2010-1-1':'2010-12-31'].plot(grid=True)


# In[11]:


dfG6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A2C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A2') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A3C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A3') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A4C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A4') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A5C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A5') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A6C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A6') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A1C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A2C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A2') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A3C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A3') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A4C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A4') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A5C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A5') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A6C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A6') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']


# In[12]:


df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31'].plot(grid=True)
df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31'].plot(grid=True)


# In[13]:


plt.title("Jan 2010 sales G6 ")
plt.xlabel("date Y-DD-MM")
plt.ylabel("cum sales (K)" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5
sns.scatterplot(dfG6A1C1.index,dfG6A1C1.CumSales,color = "red", label='A1 C1',  marker='s',s=100)
sns.scatterplot(dfG6A2C1.index,dfG6A2C1.CumSales,color = "blue", label='A2 C1',  marker='s',s=100)
sns.scatterplot(dfG6A3C1.index,dfG6A3C1.CumSales,color = "deepskyblue", label='A3 C1',  marker='s',s=100)
sns.scatterplot(dfG6A4C1.index,dfG6A4C1.CumSales,color = "darkorange", label='A4 C1',  marker='s',s=100)
sns.scatterplot(dfG6A5C1.index,dfG6A5C1.CumSales,color = "lightseagreen", label='A5 C1',  marker='s',s=100)
sns.scatterplot(dfG6A6C1.index,dfG6A6C1.CumSales,color = "blueviolet", label='A6 C1',  marker='s',s=100)
sns.scatterplot(dfG6A1C2.index,dfG6A1C2.CumSales,color = "red", label='A1 C2',  marker='^',s=100)
sns.scatterplot(dfG6A2C2.index,dfG6A2C2.CumSales,color = "blue", label='A2 C2',  marker='^',s=100)
sns.scatterplot(dfG6A3C2.index,dfG6A3C2.CumSales,color = "deepskyblue", label='A3 C2',  marker='^',s=100)
sns.scatterplot(dfG6A4C2.index,dfG6A4C2.CumSales,color = "darkorange", label='A4 C2',  marker='^',s=100)
sns.scatterplot(dfG6A5C2.index,dfG6A5C2.CumSales,color = "lightseagreen", label='A5 C2',  marker='^',s=100)
sns.scatterplot(dfG6A6C2.index,dfG6A6C2.CumSales,color = "blueviolet", label='A6 C2', marker='^',s=100)
plt.show()


# In[14]:


dfG1A1C1 = df[(df['ProductGroup']=='G1') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG2A1C1 = df[(df['ProductGroup']=='G2') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG3A1C1 = df[(df['ProductGroup']=='G3') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG4A1C1 = df[(df['ProductGroup']=='G4') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG5A1C1 = df[(df['ProductGroup']=='G5') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-1-31']
dfG1A1C2 = df[(df['ProductGroup']=='G1') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG2A1C2 = df[(df['ProductGroup']=='G2') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG3A1C2 = df[(df['ProductGroup']=='G3') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG4A1C2 = df[(df['ProductGroup']=='G4') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG5A1C2 = df[(df['ProductGroup']=='G5') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']
dfG6A1C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2010-1-1':'2010-1-31']


# In[15]:


dfG1A1C1


# In[16]:


plt.title("Jan 2010 sales A1 ")
plt.xlabel("date Y-DD-MM")
plt.ylabel("cum sales (K)" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5
sns.scatterplot(dfG1A1C1.index,dfG1A1C1.CumSales,color = "red", label='G1 C1',  marker='s',s=100)
sns.scatterplot(dfG2A1C1.index,dfG2A1C1.CumSales,color = "blue", label='G2 C1',  marker='s',s=100)
sns.scatterplot(dfG3A1C1.index,dfG3A1C1.CumSales,color = "deepskyblue", label='G3 C1',  marker='s',s=100)
sns.scatterplot(dfG4A1C1.index,dfG4A1C1.CumSales,color = "darkorange", label='G4 C1',  marker='s',s=100)
sns.scatterplot(dfG5A1C1.index,dfG5A1C1.CumSales,color = "lightseagreen", label='G5 C1',  marker='s',s=100)
sns.scatterplot(dfG6A1C1.index,dfG6A1C1.CumSales,color = "blueviolet", label='G6 C1',  marker='s',s=100)
sns.scatterplot(dfG1A1C2.index,dfG1A1C2.CumSales,color = "red", label='G1 C2',  marker='^',s=100)
sns.scatterplot(dfG2A1C2.index,dfG2A1C2.CumSales,color = "blue", label='G2 C2',  marker='^',s=100)
sns.scatterplot(dfG3A1C2.index,dfG3A1C2.CumSales,color = "deepskyblue", label='G3 C2',  marker='^',s=100)
sns.scatterplot(dfG4A1C2.index,dfG4A1C2.CumSales,color = "darkorange", label='G4 C2',  marker='^',s=100)
sns.scatterplot(dfG5A1C2.index,dfG5A1C2.CumSales,color = "lightseagreen", label='G5 C2',  marker='^',s=100)
sns.scatterplot(dfG6A1C2.index,dfG6A1C2.CumSales,color = "blueviolet", label='G6 C2', marker='^',s=100)
plt.show()


# In[17]:


dfy11G1A1C1 = df[(df['ProductGroup']=='G1') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G2A1C1 = df[(df['ProductGroup']=='G2') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G3A1C1 = df[(df['ProductGroup']=='G3') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G4A1C1 = df[(df['ProductGroup']=='G4') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G5A1C1 = df[(df['ProductGroup']=='G5') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-31']
dfy11G1A1C2 = df[(df['ProductGroup']=='G1') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']
dfy11G2A1C2 = df[(df['ProductGroup']=='G2') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']
dfy11G3A1C2 = df[(df['ProductGroup']=='G3') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']
dfy11G4A1C2 = df[(df['ProductGroup']=='G4') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']
dfy11G5A1C2 = df[(df['ProductGroup']=='G5') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']
dfy11G6A1C2 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C2')].loc['2011-1-1':'2011-1-31']


# In[18]:


dfy11G1A1C1


# In[19]:


plt.title("Jan 2011 sales A1 C1")
plt.xlabel("date Y-DD-MM")
plt.ylabel("cum sales (K)" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5
sns.scatterplot(dfy11G1A1C1.index,dfy11G1A1C1.CumSales,color = "red", label='G1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A2C1.index,dfy11G6A2C1.CumSales,color = "blue", label='A2 C1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A3C1.index,dfy11G6A3C1.CumSales,color = "deepskyblue", label='A3 C1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A4C1.index,dfy11G6A4C1.CumSales,color = "darkorange", label='A4 C1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A5C1.index,dfy11G6A5C1.CumSales,color = "lightseagreen", label='A5 C1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A6C1.index,dfy11G6A6C1.CumSales,color = "blueviolet", label='A6 C1',  marker='s',s=100)
# sns.scatterplot(dfy11G6A1C2.index,dfy11G6A1C2.CumSales,color = "red", label='A1 C2',  marker='^',s=100)
# sns.scatterplot(dfy11G6A2C2.index,dfy11G6A2C2.CumSales,color = "blue", label='A2 C2',  marker='^',s=100)
# sns.scatterplot(dfy11G6A3C2.index,dfy11G6A3C2.CumSales,color = "deepskyblue", label='A3 C2',  marker='^',s=100)
# sns.scatterplot(dfy11G6A4C2.index,dfy11G6A4C2.CumSales,color = "darkorange", label='A4 C2',  marker='^',s=100)
# sns.scatterplot(dfy11G6A5C2.index,dfy11G6A5C2.CumSales,color = "lightseagreen", label='A5 C2',  marker='^',s=100)
# sns.scatterplot(dfy11G6A6C2.index,dfy11G6A6C2.CumSales,color = "blueviolet", label='A6 C2', marker='^',s=100)
plt.show()


# In[20]:


df1Y10G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-1-1':'2010-12-31']
df1Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-12-31']


# In[21]:


plt.title("Jan 2010-2011 G6 A1 C1")
plt.xlabel("date Y-MM")
plt.ylabel("cum sales (K)" )
sns.set_style('darkgrid')
sns.scatterplot(df1Y10G6A1C1.index,df1Y10G6A1C1.CumSales,color = "blue",label='A1 2010')
sns.scatterplot(df1Y11G6A1C1.index,df1Y11G6A1C1.CumSales,color = "red",label='A1 2011')

plt.plot(df1Y10G6A1C1.index,df1Y10G6A1C1.CumSales,color = "blue",linewidth='1')
plt.plot(df1Y11G6A1C1.index,df1Y11G6A1C1.CumSales,color = "red",linewidth='1')
plt.show()


# # Linear reggression model 
# # Y2011-M01 G1 A1 C1

# In[22]:


dfy11G1A1C1


# In[23]:


model_y11G1A1C1=dfy11G1A1C1.reset_index(drop=True)
model_y11G1A1C1


# dfy11G6A1C1=dfy11G6A1C1.sort_index()
# dfy11G6A1C1=dfy11G6A1C1.set_index('column_name')
# dfy11G6A1C1.index = range(1, 31, 1) #a range starting at one ending at 30 with a stepsize of 1.
# dfy11G6A1C1=dfy11G6A1C1.sort_values(by='column_name')

# In[24]:


x=np.array(model_y11G1A1C1.index)
x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22]
x=np.array(x)
x


# In[25]:


y=np.array(model_y11G1A1C1.CumSales)
y=[10794.,23567.,30763.,57388.,78436.,90309.,118733.,124130.,156152.,181337., 192491., 194650., 212640., 221995.,
       256355., 285678., 287837., 314821., 329932., 354758., 362673.,
       369149., 407287.]
y=np.array(y)
y


# In[26]:


plt.scatter(dfy11G1A1C1.index,y)
plt.grid()
plt.show()


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[28]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)
x,y


# ## test set

# In[29]:


model5 = LinearRegression()
model5.fit(x[:5],y[:5])
model3 = LinearRegression()
model3.fit(x[:3],y[:3])
model4 = LinearRegression()
model4.fit(x[:4],y[:4])


# In[30]:


print("date : ",dfy11G1A1C1.index[22],"","predict 3: ",model3.predict([[22]]))
print("date : ",dfy11G1A1C1.index[22],"","predict 4: ",model4.predict([[22]]))
print("date : ",dfy11G1A1C1.index[22],"","predict 5: ",model5.predict([[22]]))


# In[31]:


print("date : ",dfy11G1A1C1.index[22],',',"data CumSales : ",dfy11G1A1C1.CumSales[22])


# In[32]:


prd3 = model3.predict(x)
prd4 = model4.predict(x)
prd5 = model5.predict(x)
prd5 


# ## graph

# In[33]:


plt.title("Jan 2011 Cum sales G6 ")
plt.xlabel("date")
plt.ylabel("cum sales" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5

plt.scatter(x,dfy11G1A1C1.CumSales,color='blue', label='data')
plt.plot(x,prd3,color="green",marker='s', label='predict3')
plt.plot(x,prd4,color="violet",marker='s', label='predict4')
plt.plot(x,prd5,color="red",marker='s', label='predict5')

plt.legend();
plt.show()


# In[34]:


plt.title("Jan 2011 Cum sales G6 ")
plt.xlabel("date")
plt.ylabel("cum sales" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5

#sns.scatterplot(x,dfy11G6A1C1.CumSales,color = "blue", label='data',  marker='s',s=100)
#sns.scatterplot(x,prd,color = "red", label='predict',  marker='s',s=100)

plt.plot(x,dfy11G1A1C1.CumSales,color = "blue",linewidth='1', label='data')
#plt.plot(x,prd3,color = "green",linewidth='1', label='predict3')
#plt.plot(x,prd4,color = "violet",linewidth='1', label='predict4')
plt.plot(x,prd5,color = "red",linewidth='1', label='predict5')

plt.legend();
plt.show()


# ## score

# In[35]:


y=dfy11G1A1C1.CumSales
y_pred3= prd3
# print("3 :")
# print("MAE = ",mean_absolute_error(y,y_pred3))
# print("MSE = ",mean_squared_error(y,y_pred3))
# print("r2  = ",r2_score(y,y_pred3))
# y_pred4= prd4
# print("4 :")
# print("MAE = ",mean_absolute_error(y,y_pred4))
# print("MSE = ",mean_squared_error(y,y_pred4))
# print("r2  = ",r2_score(y,y_pred4))

y_pred5= prd5
print("5 :")
print("MAE = ",mean_absolute_error(y,y_pred5))
print("MSE = ",mean_squared_error(y,y_pred5))
print("r2  = ",r2_score(y,y_pred5))


# # Linear reggression model 
# # Y2011-M01 G2 A1 C1

# In[36]:


dfy11G2A1C1


# In[37]:


model_y11G2A1C1=dfy11G2A1C1.reset_index(drop=True)
model_y11G2A1C1


# In[38]:


x=np.array(model_y11G2A1C1.index)
x=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]
x=np.array(x)
x


# In[39]:


y=np.array(model_y11G2A1C1.CumSales)
y=[  53969.,  112076.,  169643.,  218575.,  294671.,  361952.,
        387497.,  425275.,  490397.,  540768.,  590779.,  658780.,
        716347.,  760961.,  793342.,  865121.,  922148.,  973958.,
       1033324., 1093769., 1122912., 1187675., 1222935., 1259634.,
       1318640.]
y=np.array(y)
y


# In[40]:


plt.scatter(dfy11G2A1C1.index,y)
plt.grid()
plt.show()


# In[41]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)


# In[42]:


model = LinearRegression()
model.fit(x[:5],y[:5])


# In[43]:


print("date : ",dfy11G2A1C1.index[22],"","predict : ",model.predict([[24]]))
print("date : ",dfy11G2A1C1.index[22],',',"data CumSales : ",dfy11G2A1C1.CumSales[22])


# In[44]:


prd = model.predict(x)


# In[45]:


plt.title("Jan 2011 Cum sales G2 ")
plt.xlabel("date")
plt.ylabel("cum sales" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5

plt.scatter(x,dfy11G2A1C1.CumSales,color='blue', label='data')
plt.plot(x,prd,color="red",marker='s', label='predict')

plt.legend();
plt.show()


# In[46]:


plt.title("Jan 2011 Cum sales G2 ")
plt.xlabel("date")
plt.ylabel("cum sales" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5

#sns.scatterplot(x,dfy11G6A1C1.CumSales,color = "blue", label='data',  marker='s',s=100)
#sns.scatterplot(x,prd,color = "red", label='predict',  marker='s',s=100)

plt.plot(x,dfy11G2A1C1.CumSales,color = "blue",linewidth='1', label='data')
plt.plot(x,prd,color = "red",linewidth='1', label='predict')

plt.legend();
plt.show()


# In[47]:


y=dfy11G2A1C1.CumSales
y_pred= prd
print("MAE : ",mean_absolute_error(y,y_pred))
print("MSE : ",mean_squared_error(y,y_pred))
print("r2  : ",r2_score(y,y_pred))


# # pre year
# 

# In[48]:


df1Y11G6A1C1


# In[49]:


df1Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-12-30']


# In[50]:


dfD15M1Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-1':'2011-1-15']
dfD15M2Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-2-1':'2011-2-15']
dfD15M3Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-3-1':'2011-3-15']
dfD15M4Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-4-1':'2011-4-15']
dfD15M5Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-5-1':'2011-5-15']
dfD15M6Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-6-1':'2011-6-15']
dfD15M7Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-7-1':'2011-7-14']
dfD15M8Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-8-1':'2011-8-15']
dfD15M9Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-9-1':'2011-9-15']
dfD15M10Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-10-1':'2011-10-15']
dfD15M11Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-11-1':'2011-11-15']
dfD15M12Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-12-1':'2011-12-15']
#print(dfD15M12Y11G6A1C1.index)
#print(dfD15M12Y11G6A1C1.CumSales)
dfD15M12Y11G6A1C1
print(dfD15M1Y11G6A1C1.pivot_table(index=['DayKey'],values='CumSales',aggfunc='sum',margins=True))
print(dfD15M2Y11G6A1C1.pivot_table(index=['DayKey'],values='CumSales',aggfunc='sum',margins=True))
print(dfD15M3Y11G6A1C1.pivot_table(index=['DayKey'],values='CumSales',aggfunc='sum',margins=True))


# In[51]:


x_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y_cumsale=[15866351.0,14580164.0,15600745.0,11702183.0,13416606.0,14626325.0,11805992.0,14470051.0,16898338.0,16730129.0,13204180.0,14641683.0]


# In[52]:


y_cumsale


# In[53]:


plt.title("2011 G6 A1 C1")
plt.xlabel("date Month")
plt.ylabel("cum sales " )
sns.set_style('darkgrid')

sns.scatterplot(x_date,y_cumsale,color = "red",label='A1 2011')

plt.plot(x_date,y_cumsale,color = "red",linewidth='1')
plt.show()


# In[54]:


model_1Y11G6A1C1=df1Y11G6A1C1.reset_index(drop=True)
model_1Y11G6A1C1


# In[55]:


plt.xlabel("date Y-MM")
plt.ylabel("cum sales (K)" )
sns.set_style('darkgrid')

sns.scatterplot(model_1Y11G6A1C1.index,model_1Y11G6A1C1.CumSales,color = "red",label='A1 2011')

plt.plot(model_1Y11G6A1C1.index,model_1Y11G6A1C1.CumSales,color = "red",linewidth='1')
plt.show()


# # Polynomial regression year

# In[56]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy


# In[57]:


x=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]
poly = numpy.poly1d(numpy.polyfit(x[:5], y[:5], 2))

myline = numpy.linspace(1, 24, 1318640)


# In[58]:


plt.title("Jan 2011 Cum sales G2 ")
plt.xlabel("date")
plt.ylabel("cum sales" )
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']=15,5

plt.scatter(x,dfy11G2A1C1.CumSales,color='blue', label='data')
plt.plot(x,prd,color="red",marker='s', label='linear')
plt.plot(myline,poly(myline),color="green",linewidth='1', label='poly')

plt.legend();
plt.show()


# In[85]:


x_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y_cumsale=[15866351.0,14580164.0,15600745.0,11702183.0,13416606.0,14626325.0,11805992.0,14470051.0,16898338.0,16730129.0,13204180.0,14641683.0]
poly = numpy.poly1d(numpy.polyfit(x_date[:], y_cumsale[:], 2))
linear = numpy.poly1d(numpy.polyfit(x_date[:], y_cumsale[:], 1))

myline = numpy.linspace(1, 12, 14641683)
plt.title("2011 G6 A1 C1")
plt.xlabel("date Month")
plt.ylabel("cum sales " )
sns.set_style('darkgrid')

# sns.scatterplot(x_date,y_cumsale,color = "red",label='A1 2011')
plt.plot(myline,poly(myline),color="green",linewidth='1', label='poly')
plt.plot(myline,linear(myline),color="green",linewidth='1', label='linear')

plt.plot(x_date,y_cumsale,color = "red",linewidth='1')
plt.show()
print("linear")
print("MAE : ",mean_absolute_error(y, linear(x)))
print("MSE : ",mean_squared_error(y, linear(x)))
print("r2  : ",r2_score(y, poly(x)))
print("\npoly")
print("MAE : ",mean_absolute_error(y, poly(x)))
print("MSE : ",mean_squared_error(y, poly(x)))
print("r2  : ",r2_score(y, poly(x)))


# In[60]:


import numpy
import matplotlib.pyplot as plt

x=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]
y=[  53969.,  112076.,  169643.,  218575.,  294671.,  361952.,
        387497.,  425275.,  490397.,  540768.,  590779.,  658780.,
        716347.,  760961.,  793342.,  865121.,  922148.,  973958.,
       1033324., 1093769., 1122912., 1187675., 1222935., 1259634.,
       1318640.]

mymodel = numpy.poly1d(numpy.polyfit(x[:5], y[:5], 2))

myline = numpy.linspace(1, 24, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline), label='poly')
plt.plot(x,prd,color="red",marker='s', label='linear')
plt.show()
speed = mymodel(22)
print(speed)


# In[61]:


y=dfy11G2A1C1.CumSales
y_pred= prd
print("linear")
print("MAE : ",mean_absolute_error(y,y_pred))
print("MSE : ",mean_squared_error(y,y_pred))
print("r2  : ",r2_score(y,y_pred))
print("\npoly")
print("MAE : ",mean_absolute_error(y, mymodel(x)))
print("MSE : ",mean_squared_error(y, mymodel(x)))
print("r2  : ",r2_score(y, mymodel(x)))


# # Years

# In[62]:


dfD15M1Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-1-15']
dfD15M2Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-2-15']
dfD15M3Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-3-15']
dfD15M4Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-4-15']
dfD15M5Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-5-15']
dfD15M6Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-6-15']
dfD15M7Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-7-16']
dfD15M8Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-8-15']
dfD15M9Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-9-15']
dfD15M10Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-10-15']
dfD15M11Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-11-15']
dfD15M12Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-12-15']


# In[63]:


dfD15M7Y11G6A1C1


# In[64]:


x11_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y11_cumsale=[dfD15M1Y11G6A1C1.CumSales,dfD15M2Y11G6A1C1.CumSales,dfD15M3Y11G6A1C1.CumSales,dfD15M4Y11G6A1C1.CumSales
            ,dfD15M5Y11G6A1C1.CumSales,dfD15M6Y11G6A1C1.CumSales,dfD15M7Y11G6A1C1.CumSales,dfD15M8Y11G6A1C1.CumSales
            ,dfD15M9Y11G6A1C1.CumSales,dfD15M10Y11G6A1C1.CumSales,dfD15M11Y11G6A1C1.CumSales,dfD15M12Y11G6A1C1.CumSales]


# In[65]:


dfD15M1Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-1-15']
dfD15M2Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-2-15']
dfD15M3Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-3-15']
dfD15M4Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-4-15']
dfD15M5Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-5-15']
dfD15M6Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-6-15']
dfD15M7Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-7-16']
dfD15M8Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-8-15']
dfD15M9Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-9-15']
dfD15M10Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-10-15']
dfD15M11Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-11-15']
dfD15M12Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-12-15']


# In[66]:


df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-7-14']


# In[67]:


df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-7-16']


# In[68]:


x13_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y13_cumsale=[dfD15M1Y13G6A1C1.CumSales,dfD15M2Y13G6A1C1.CumSales,dfD15M3Y13G6A1C1.CumSales,dfD15M4Y13G6A1C1.CumSales
            ,dfD15M5Y13G6A1C1.CumSales,dfD15M6Y13G6A1C1.CumSales,dfD15M7Y13G6A1C1.CumSales,dfD15M8Y13G6A1C1.CumSales
            ,dfD15M9Y13G6A1C1.CumSales,dfD15M10Y13G6A1C1.CumSales,dfD15M11Y13G6A1C1.CumSales,dfD15M12Y13G6A1C1.CumSales]


# In[69]:


dfD15M1Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-1-15']
dfD15M2Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-2-15']
dfD15M3Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-3-15']
dfD15M4Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-4-15']
dfD15M5Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-5-15']
dfD15M6Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-6-15']
dfD15M7Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-7-16']
dfD15M8Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-8-15']
dfD15M9Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-9-15']
dfD15M10Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-10-15']
dfD15M11Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-11-15']
dfD15M12Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-12-15']


# In[70]:


dfD15M7Y14G6A1C1


# In[71]:


x14_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y14_cumsale=[dfD15M1Y14G6A1C1.CumSales,dfD15M2Y14G6A1C1.CumSales,dfD15M3Y14G6A1C1.CumSales,dfD15M4Y14G6A1C1.CumSales
            ,dfD15M5Y14G6A1C1.CumSales,dfD15M6Y14G6A1C1.CumSales,dfD15M7Y14G6A1C1.CumSales,dfD15M8Y14G6A1C1.CumSales
            ,dfD15M9Y14G6A1C1.CumSales,dfD15M10Y14G6A1C1.CumSales,dfD15M11Y14G6A1C1.CumSales,dfD15M12Y14G6A1C1.CumSales]


# In[72]:


dfD15M1Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-1-15']
dfD15M2Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-2-15']
dfD15M3Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-3-15']
dfD15M4Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-4-15']
dfD15M5Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-5-15']
dfD15M6Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-6-15']
dfD15M7Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-7-16']
dfD15M8Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-8-15']
dfD15M9Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-9-15']
dfD15M10Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-10-15']
dfD15M11Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-11-15']
dfD15M12Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-12-15']


# In[73]:


dfD15M7Y15G6A1C1


# In[74]:


x15_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y15_cumsale=[dfD15M1Y15G6A1C1.CumSales,dfD15M2Y15G6A1C1.CumSales,dfD15M3Y15G6A1C1.CumSales,dfD15M4Y15G6A1C1.CumSales
            ,dfD15M5Y15G6A1C1.CumSales,dfD15M6Y15G6A1C1.CumSales,dfD15M7Y15G6A1C1.CumSales,dfD15M8Y15G6A1C1.CumSales
            ,dfD15M9Y15G6A1C1.CumSales,dfD15M10Y15G6A1C1.CumSales,dfD15M11Y15G6A1C1.CumSales,dfD15M12Y15G6A1C1.CumSales]


# In[75]:


dfD15M1Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-1-15']
dfD15M2Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-2-15']
dfD15M3Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-3-15']
dfD15M4Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-4-16']
dfD15M5Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-5-16']
dfD15M6Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-6-15']
dfD15M7Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-7-16']
dfD15M8Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-8-15']
dfD15M9Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-9-15']
dfD15M10Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-10-15']
dfD15M11Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-11-15']
dfD15M12Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-12-15']


# In[76]:


dfD15M7Y16G6A1C1


# In[77]:


x16_date=[1,2,3,4,5,6,7,8,9,10,11,12]
y16_cumsale=[dfD15M1Y16G6A1C1.CumSales,dfD15M2Y16G6A1C1.CumSales,dfD15M3Y16G6A1C1.CumSales,dfD15M4Y16G6A1C1.CumSales
            ,dfD15M5Y16G6A1C1.CumSales,dfD15M6Y16G6A1C1.CumSales,dfD15M7Y16G6A1C1.CumSales,dfD15M8Y16G6A1C1.CumSales
            ,dfD15M9Y16G6A1C1.CumSales,dfD15M10Y16G6A1C1.CumSales,dfD15M11Y16G6A1C1.CumSales,dfD15M12Y16G6A1C1.CumSales]
x16=np.array(x16_date)
y16=np.array(y16_cumsale)
x16=x16.reshape(-1,1)
y16=y16.reshape(-1,1)
model = LinearRegression()
model.fit(x16[:],y16[:])
prd16 = model.predict(x16)


# In[ ]:





# In[78]:


plt.title("day 15 G6 A1 C1")
plt.xlabel("date Month")
plt.ylabel("cum sales " )
sns.set_style('darkgrid')

sns.scatterplot(x11_date,y11_cumsale,color = "red",label='A1 2011')
sns.scatterplot(x13_date,y13_cumsale,color = "blue",label='A1 2013')
sns.scatterplot(x14_date,y14_cumsale,color = "orange",label='A1 2014')
sns.scatterplot(x15_date,y15_cumsale,color = "purple",label='A1 2015')
sns.scatterplot(x16_date,y16_cumsale,color = "teal",label='data 2016')
plt.plot(x16,prd16,color="teal",marker='^', label='pre 16')

plt.plot(x16_date,y16_cumsale,color = "teal",linewidth='1')
plt.plot(x15_date,y15_cumsale,color = "purple",linewidth='1')
plt.plot(x14_date,y14_cumsale,color = "orange",linewidth='1')
plt.plot(x13_date,y13_cumsale,color = "blue",linewidth='1')
plt.plot(x11_date,y11_cumsale,color = "red",linewidth='1')
plt.show()


# In[79]:


df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-7-16']


# In[80]:


dfD15M12Y10G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2010-7-15']
dfD15M12Y11G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2011-7-16']
dfD15M12Y12G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2012-7-16']
dfD15M12Y13G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2013-7-16']
dfD15M12Y14G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2014-7-16']
dfD15M12Y15G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2015-7-16']
dfD15M12Y16G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2016-7-16']
dfD15M12Y17G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2017-7-15']
dfD15M12Y18G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2018-7-16']
dfD15M12Y19G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2019-7-15']
dfD15M12Y20G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2020-7-16']
dfD15M12Y21G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2021-7-16']
#dfD15M12Y22G6A1C1 = df[(df['ProductGroup']=='G6') & (df['Area']=='A1') & (df['Channel']=='C1')].loc['2022-7-17']
x_year=[10,11,12,13,14,15,16,17,18,19,20,21]
y_cumsale=[dfD15M12Y10G6A1C1.CumSales,dfD15M12Y11G6A1C1.CumSales,dfD15M12Y12G6A1C1.CumSales,dfD15M12Y13G6A1C1.CumSales
          ,dfD15M12Y14G6A1C1.CumSales,dfD15M12Y15G6A1C1.CumSales,dfD15M12Y16G6A1C1.CumSales,dfD15M12Y17G6A1C1.CumSales
          ,dfD15M12Y18G6A1C1.CumSales,dfD15M12Y19G6A1C1.CumSales,dfD15M12Y20G6A1C1.CumSales,dfD15M12Y21G6A1C1.CumSales]
x=np.array(x_year)
y=np.array(y_cumsale)
x=x.reshape(-1,1)
y=y.reshape(-1,1)
model = LinearRegression()
model.fit(x[:4],y[:4])
prd = model.predict(x)


# In[81]:


plt.title(" G6 A1 C1")
plt.xlabel("year 20xx")
plt.ylabel("cum sales " )
sns.set_style('darkgrid')

plt.plot(x_year,prd,color="blue",marker='^')
plt.plot(x_year,y_cumsale,color="red",marker='s')

plt.show()


# In[82]:


y_pred= prd
print("linear")
print("MAE : ",mean_absolute_error(y_cumsale,y_pred))
print("MSE : ",mean_squared_error(y_cumsale,y_pred))
print("r2  : ",r2_score(y_cumsale,y_pred))


# In[ ]:




