import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv("dataset/major_league_baseball_players.csv")