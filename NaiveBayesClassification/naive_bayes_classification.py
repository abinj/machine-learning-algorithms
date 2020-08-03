import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('pacific.csv')
print(data.head(6))

data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes
print(data.head())
sns.countplot(data['Status'], label="Count")
plt.show()

# Data wrangling
