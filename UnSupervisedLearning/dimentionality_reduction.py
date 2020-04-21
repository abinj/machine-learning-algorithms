import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

existing_df = pd.read_csv('trans_us.csv', index_col = 0, thousands=',')
existing_df.index.names = ['stations']
existing_df.columns.names = ['months']
existing_df = existing_df.fillna(15)
print(existing_df.head())

pca = PCA(n_components=2)
pca.fit(existing_df)
existing_2d = pca.transform(existing_df)
existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = existing_df.index
existing_df_2d.columns = ['PC1', 'PC2']
print(existing_df_2d.head())

print(pca.explained_variance_ratio_)
