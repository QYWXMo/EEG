import pandas as pd
import numpy as np

from dataloader import data_loader

# dataloader
path = './data.mat'
df = data_loader(path)
df.dropna()
df_row = df[13]
print(df_row.max())
print(df_row.min())
print(df_row.var())
print(df_row.mean())