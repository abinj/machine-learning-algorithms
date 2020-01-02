import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

boston['MEDV'] = boston_dataset.target

boston.isnull().sum()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']
print(boston['MEDV'].count())

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

plt.show()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
Y = boston['MEDV']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=5)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

model = LinearRegression()
model.fit(train_x, train_y)

#Model Evaluation
predict = model.predict(train_x)
rmse = (np.sqrt(mean_squared_error(train_y, predict)))
r2 = r2_score(train_y, predict)

print("The model performance for training set")
print("RMSE is: {}".format(rmse))
print("R2 value is {}".format(r2))

print("Model performace for testing set")
test_predict = model.predict(test_x)
rmse = (np.sqrt(mean_squared_error(test_y, test_predict)))
r2 = r2_score(test_y, test_predict)

print("Test set RMSE is: {}".format(rmse))
print("Test set r2 value is: {}".format(r2))

