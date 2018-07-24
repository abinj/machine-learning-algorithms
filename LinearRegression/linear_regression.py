import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

from sklearn import linear_model

df = pd.read_csv("LinearRegression/linear_regression_df.csv")
df.columns = ['X', 'Y']
plt.scatter(x=df['X'], y=df['Y'])
plt.ylabel('Response')
plt.xlabel('Explanatory')

linear = linear_model.LinearRegression()
trainX = np.asarray(df.X[20:len(df.X)]).reshape(-1, 1)
trainY = np.asarray(df.Y[20:len(df.Y)]).reshape(-1, 1)
testX = np.asarray(df.X[:20]).reshape(-1, 1)
testY = np.asarray(df.Y[:20]).reshape(-1, 1)
linear.fit(trainX, trainY)
r2_value = linear.score(trainX, trainY)
print('Coefficient: \n', linear.coef_)
print("Intercept: \n", linear.intercept_)
print("R^2value: \n", r2_value)
predicted = linear.predict(testX)
print(predicted)

plt.plot(testX, predicted)
plt.show()
