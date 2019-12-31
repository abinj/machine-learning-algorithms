import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/home/abin/my_works/github_works/machine-learning-algorithms/LogisticRegression/dataset/iris.csv")
print(data.head())

print("\n column names:-")
print(data.columns)

#Encode labels with value between 0 and n_classes-1
encode = LabelEncoder()
data.Species = encode.fit_transform(data.Species)

print(data.head())

#Split train and test data
train, test = train_test_split(data, test_size=0.2, random_state=0)

print("\n Shape of training data:- ", train.shape)
print("\n Shape of test data:- ", test.shape)

#Separate the dependent and independent variables
train_x = train.drop(columns=['Species'], axis=1)
train_y = train['Species']


test_x = test.drop(columns=['Species'], axis=1)
test_y = test['Species']

#Train the model
model = LogisticRegression(multi_class='auto')
model.fit(train_x, train_y)

predict = model.predict(test_x)

print("\n Predicted values on test data", encode.inverse_transform(predict))

print("\n\nAccuracy score on test data: ")
print(accuracy_score(test_y, predict))

report = classification_report(test_y, predict)
print("\nClassification Report: ", report)

