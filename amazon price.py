

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
os.chdir("D:")
df = pd.read_csv('price dataset.csv')
df.head()
df.isnull().sum()
df.info()
df.describe()
df.shape
df['symbol'].nunique()    #number of 



df1 = df[df['symbol']=='AMZN']
df1.head(5)

df1.describe()
np.round(df1.median() , 2)
## as we can see mean > median , so it is right skewed


colors = ['yellow','black']
sns.set(palette = colors , font = 'Arial'
        ,style = 'white' , rc = {'axes.facecolor':'whitesmoke','figure.facecolor':'whitesmoke'})
sns.palplot(colors , size = 2)

fig = plt.figure(figsize = (20,8))
ax = sns.lineplot(data = df1 , x = 'date' , y = 'open')
ax = sns.lineplot(data = df1 , x  = 'date' , y = 'close' , color = colors[1])

for s in ['left','right','top','bottom']:
    ax.spines[s].set_visible(False)
    
plt.title("AMAZON Stock value changes since 2010", size = 20 , weight = 'bold')

fig=plt.figure(figsize=(20,8))
ax=sns.lineplot(data=df1, x='date',y='volume',color='yellow')
#ax=sns.lineplot(data=df1, x='date',y='close', color=colors[1]);
for s in ['left','right','top','bottom']:
    ax.spines[s].set_visible(False)
plt.title("Google Stock volume", size=20, weight='bold')


##Unovariated Analysis - error 

df1=sns.PairGrid(df,hue="volume")
df1.map(plt.scatter)


# we need to predict the closing price of the stock, lets us consider 'Close' feature as the Target variable.
sns.pairplot(df1,corner=True)
df1.corr()['close']
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()

#normality Check


##test & split
X = df1[['open','volume']]
y=df1['close']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=False, random_state=42)

#Nirmalizing the 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)


#Basic linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error ,mean_absolute_error

model= LinearRegression()

model.fit(X_train , y_train)
 
pred = model.predict(X_test)
pred.mean()
score = np.round(model.score(X_test , y_test),2)*100
score
r2 = np.round(r2_score(y_test , pred),2)
mse = np.round(mean_squared_error(y_test,pred),2)
mae = np.round(mean_squared_error(y_test,pred),2)
r2,mse,mae


fig = plt.figure(figsize=(15,8))
p = pd.Series(pred,index = y_test.index)
plt.plot(y_test)
plt.plot(p)
plt.legend(['y_test','predicted'])

plt.title("Compare test and predicted values", size=20, weight='bold')
plt.text(x=800000, y=600,s='Accuracy score : {} %'.format(score))
plt.text(x=800000, y=580,s='R2 Score : {}'.format(r2))
plt.text(x=800000, y=560,s='Mean Squared error : {}'.format(mse))
plt.text(x=800000, y=540,s='Mean Absolute error : {}'.format(mae))


df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
df.head()

print('Mean Absolute Error:', mean_absolute_error(y_test, pred))  
print('Mean Squared Error:', mean_squared_error(y_test, pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))

######logistic Regression
##continuous data so can't apply
from sklearn.linear_model import LogisticRegression 
m = LogisticRegression(random_state = 0)
m.fit(X_train,y_train)


#######Lasso 
from sklearn.linear_model import Lasso
l = Lasso(normalize = True)
l.fit(X_train , y_train)
l_pred = l.predict(X_test)
l_pred.mean()
score_lasso=l.score(X_test, y_test)
score_lasso
df = pd.DataFrame({'Actual': y_test, 'Predicted': l_pred})
df.head()


############decision tree regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train,y_train)
y_predict_dtr = dtr.predict(X_test) 

score_dtr=dtr.score(X_test, y_test)
print(score_dtr)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict_dtr})
df.head()


#############



import numpy as np
red = plt.scatter(np.arange(0,80,5),pred[0:80:5],color = "red")
green = plt.scatter(np.arange(0,80,5),l_pred[0:80:5],color = "green")
blue = plt.scatter(np.arange(0,80,5),y_predict_dtr[0:80:5],color = "blue")
black = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "black")

plt.title("Comparison of Regression Algorithms")
plt.xlabel("Index of Candidate")
plt.ylabel("Chance of Admit")
plt.legend((red,green,blue,black),('LR','Lasso','DTR', 'REAL'))
plt.show()



















