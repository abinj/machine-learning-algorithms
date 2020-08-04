import pandas as pd
import pathlib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
print(pathlib.Path().absolute())
df = pd.read_csv(str(pathlib.Path().absolute()) + "/dataset/diabetes_data.csv")
print(df.head())
print(df.shape)

# Split the data
X = df.drop(columns=['diabetes'])
print(X.head())

y = df['diabetes'].values
print(y[0:6])


# Split the dataset into train and test data
# stratify=y, makes sure that the training split
# represent the proportion of each value in the y variable. For example, if the dtaset has 25% of patients
#  with diabetes and 75% no diabetes, then our random split also follow the same proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

# Train the model
# n_neighbors = 3, a new data point is labelled with by majority from the 3 nearest points.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# Testing the model
print(knn.predict(X_test)[0:5])

# Check accuracy of our trained model on the test data
print(knn.score(X_test, y_test))

# Cross-validation
knn_cv = KNeighborsClassifier(n_neighbors=3)

# Train model with cv = 5
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

print(cv_scores)
print('cv scores mean:{}'.format(np.mean(cv_scores)))


# Hypertuning model parameters
knn_gsearch = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 25)}

knn_gscv = GridSearchCV(knn_gsearch, param_grid, cv=5)

knn_gscv.fit(X, y)

# Check top performing n_neighbors value
print(knn_gscv.best_params_)

# Check mean scores  for the top performing value of n_neighbors
print(knn_gscv.best_score_)


