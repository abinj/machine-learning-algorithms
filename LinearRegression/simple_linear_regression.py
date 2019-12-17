import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Read dataset
df = pd.read_csv("dataset/grey_kangaroos.csv")
df.columns = ['X', 'Y']

# Split the dataset for training and test data
trainX = np.asarray(df.X[20:len(df.X)]).reshape(-1, 1)
trainY = np.asarray(df.Y[20:len(df.Y)]).reshape(-1, 1)
testX = np.asarray(df.X[:20]).reshape(-1, 1)
testY = np.asarray(df.Y[:20]).reshape(-1, 1)

# Model initialization
model = linear_model.LinearRegression()
# Fit the data
model.fit(trainX, trainY)
# Predict
predicted = model.predict(testX)

# Model evaluation
r2_value = model.score(trainX, trainY)
# Root Mean Squared Deviation
rmsd = np.sqrt(mean_squared_error(testY, predicted))

# Printing values
print("Slope(b1): \n", model.coef_)
print("Y Intercept(b0): \n", model.intercept_)
#Lower the rmse, model best fit
print("Root Mean Square Deviation: \n", rmsd)
#Higher the R^2, model best fit
print("R^2 value: \n", r2_value)
print("Prediction over test data: \n", predicted)

plt.scatter(x=df['X'], y=df['Y'])
plt.ylabel('nasal width (mm ¥ 10)')
plt.xlabel('nasal length (mm ¥10)')
plt.plot(testX, predicted)
plt.show()
