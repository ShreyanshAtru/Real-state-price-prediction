# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:04:13 2021

@author: HP
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy.stats import shapiro 
from sklearn.model_selection import train_test_split
os.chdir('D:')
df = pd.read_csv('Real estate.csv')
df.info()
df.describe()
df.isnull().sum()
df.head(4)

columns_to_drop = ['transaction date','No']
df = df.drop(columns_to_drop , axis = 1)
df.info()

##Data visualization -->
##1. histogram of house age 
plt.hist(df['house age'])
plt.title('Histogram of houses age ')
plt.xlabel('House age')
plt.ylabel('Frequency')

##2. box plot of house price of unit area 
pricestate = sns.boxplot(df['house price of unit area'])

##3. Histogram of latitude
plt.hist(df['latitude'])

##4. Histogram of longitude 
plt.hist(df['longitude'])

plt.title('Histogram of longitude ')
plt.xlabel('longitude')
plt.ylabel('Frequency')

##5. histogram of MRT STation
plt.hist(df['distance to the nearest MRT station'])

plt.title('Histogram of distance to the nearest MRT station  ')
plt.xlabel('distance to the nearest MRT station')
plt.ylabel('Frequency')

## 6 .density plot 
df.longitude.plot.density(color='green')
plt.title('Density plot for longitude')
plt.show()

df['house age'].plot.density(color='green')
plt.title('Density plot for house age')
plt.show()

df['distance to the nearest MRT station'].plot.density(color = 'green')
plt.title('distance to the nearest MRT station')

def normality_test(df):
  stat, p_value = shapiro(df)    #Shapiro-Wilk test
  alpha = 0.05

  if p_value > alpha:
    print('Normality test: Gaussian')
  else:
    print('Normality test: Non Gaussian') 
normality_test(df)

from scipy.stats import levene, shapiro
int_cols=df.select_dtypes(exclude='object').columns.to_list()

for i in int_cols:
    _, p_value=shapiro(df[i])
    if p_value<0.05:
        print("Feature {} is normaly distributed".format(i))
    else:
        print("Feature {} is not normaly distributed".format(i))
        

#pricestate = sns.boxplot(df['house price of unit area'])
#houseage = sns.countplot(df['house age'])
#number_of_store = sns.boxplot(df['number of convenience stores'])
#Langitude = plt.scatter(df['latitude'],df['longitude'])



plt.figure(figsize=(13,5))

for feat, grd in zip(df, range(231,237)):
  plt.subplot(grd)
  sns.boxplot(y=df[feat], color='grey')
  plt.ylabel('Value')
  plt.title('Boxplot\n%s'%feat)
plt.tight_layout()


##pair   - dependenet variable is house price of unit area
sns.pairplot(df, y_vars='house price of unit area', palette = sns.set_palette(['#696969']),
             x_vars=['house age', 'distance to the nearest MRT station', 'number of convenience stores', 'latitude', 'longitude']);



correlation = df.corr()
print(correlation)


fig=plt.figure(figsize=(15,8))
colors = ['red','blue']
sns.heatmap(df.corr(), annot=True, cmap=[colors[0],colors[1]], linecolor='white', linewidth=2 )


############Test-train-split
X = df.loc[:,'house age' : 'longitude']
y = df.loc[:,'house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)



######Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error ,mean_absolute_error
lin = LinearRegression()
lin.fit(X_train,y_train)
lin_pred = lin.predict(X_test)
r2 = r2_score(y_test , lin_pred)
mse = mean_squared_error(y_test , lin_pred)
mae = mean_absolute_error(y_test,lin_pred)
print(r2 , mse,mae)

plt.figure(figsize=(10,4))

def plot_regression(real, predicted, color, title):
  plt.scatter(real, predicted, color=color)
  plt.plot([real.min(), real.max()], [real.min(), real.max()], 'k--', lw=4)
  plt.xlabel('Real Price')
  plt.ylabel('Predicted')
  plt.title(title)
  
plt.subplot(131)
plot_regression(y_test, lin_pred, 'cornflowerblue', 'Linear Regression Predictions \nTrain-test split')
plt.show()

plt.figure(figsize=(15,4))

df = pd.DataFrame({'Actual': y_test, 'Predicted': lin_pred})
df.head()

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': lin_pred})
print(df1.head(5))
########comparison btw  real and predicted
plt.plot(np.array(y_test), color='grey', label='Real')
plt.plot(lin_pred, color='cornflowerblue', label='Train-test split')
plt.xlabel('House')
plt.ylabel('Price')
plt.title('Predictions Comparison (same split)')
plt.legend(loc=4)
plt.show()





##############Decision Tree regressor 
#### Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train,y_train)
dtr_pred = dtr.predict(X_test)
dtr_r2 = r2_score(y_test , dtr_pred)
dtr_mse = mean_squared_error(y_test,dtr_pred)
print(dtr_r2,dtr_mse)


plot_regression(y_test, dtr_pred, 'cornflowerblue', 'Decision TreeRegression Predictions \nTrain-test split')
plt.show()
df = pd.DataFrame({'Actual': y_test, 'Predicted': dtr_pred})
df.tail()


##########Lasso 
from sklearn.linear_model import Lasso
l = Lasso(normalize = True)
l.fit(X_train , y_train)
l_pred = l.predict(X_test)
lasso_score = l.score(X_test, y_test)
print(lasso_score)
df = pd.DataFrame({'Actual': y_test, 'Predicted': l_pred})
df.head()
plot_regression(y_test, l_pred, 'cornflowerblue', 'Decision TreeRegression Predictions \nTrain-test split')
plt.show()




###########Random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test ,preds)
print(mae,mse)
plot_regression(y_test,preds, 'cornflowerblue', 'RAndom TreeRegression Predictions \nTrain-test split')
plt.show()


