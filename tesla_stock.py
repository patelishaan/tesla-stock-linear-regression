import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn

df = pd.read_csv('TSLA.csv')
df = df.drop(['Date'], axis=1)
#target
y = df['Close']
#features
x = df.drop(['Close','Adj Close'], axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
reg = LinearRegression()
reg.fit(x,y)
print(reg.coef_)
print(reg.intercept_)

predictions = reg.predict(xtest)
comparison = pd.DataFrame({'Predicted Values':predictions,'Actual Values':ytest})
print(comparison.head())

#plotting
#print(xtest)
#print(xtest.shape)
#print(predictions.shape)
xplot = xtest.iloc[:,0]#Open price of all rows
#print(xplot.shape)
plt.scatter(xtest.iloc[:,0], ytest, color='blue')#open vs actual close price
plt.scatter(xtest.iloc[:,0], predictions, color='red')#open vs predicted close price
plt.show()