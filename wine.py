import pandas as pd


data = pd.read_csv("data/wine-quality-white-and-red.csv")
print(data.head())
print(data.info())
print(data.describe())