import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

names = ['class_identifier', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
wine = pd.read_csv("wine.csv", names=names)
print(wine.head())

X = wine.drop('class_identifier', axis=1)
Y = wine.class_identifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =0.20)
k_range = range(1,30)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))

print(scores)

#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

classes = {1:'class 1',2:'class 2',3:'class 3'}

y_pred = knn.predict(X_test)
print(y_pred)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head(10))

df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Root Mean Squared Deviation
rmsd = np.sqrt(mean_squared_error(y_test, y_pred))      # Lower the rmse(rmsd) is, the better the fit
r2_value = r2_score(y_test, y_pred)                     # The closer towards 1, the better the fit

print("Root Mean Square Error \n", rmsd)
print("R^2 Value: \n", r2_value)
