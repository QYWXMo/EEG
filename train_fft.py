

from tools.dataloader import data_loader
from sklearn.model_selection import train_test_split
from tools.model_test import test_data
from tools.fft import draw_fft
# parameters
periods = 1


# dataloader

path = './data/data.mat'
df = data_loader(path)
df.dropna()
df_row = df[0]

train_df, test_df = train_test_split(df_row, test_size=0.2, shuffle=False) # to be continue
draw_fft(train_df)