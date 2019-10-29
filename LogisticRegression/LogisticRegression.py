import pandas as pd

# Import the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("dataset/social_network_ads.csv")
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# Split the dataset into training and test set
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size = 0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)