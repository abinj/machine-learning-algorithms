import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/home/abin/my_works/github_works/machine-learning-algorithms/LogisticRegression/dataset/loan_prediction.csv")
print(data.head())

print("\nColumn Names")
print(data.columns)

encode = LabelEncoder()
data.Loan_Status = encode.fit_transform(data.Loan_Status)

#drop the null values
data.dropna(how='any', inplace=True)

train, test = train_test_split(data, test_size=0.2, random_state=0)

train_x = train.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
test_y = test['Loan_Status']

train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)

print('shape of training data: ', train_x.shape)
print('shape of testing data : ', test_x.shape)

model = LogisticRegression(multi_class='auto')

model.fit(train_x, train_y)

predict = model.predict(test_x)

print('Predicted Values on test data ', predict)

print('\nAccuracy score on test data : ', accuracy_score(test_y, predict))

report = classification_report(test_y, predict)
print("\nClassification Report: ", report)
