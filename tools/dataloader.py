

from scipy.io import loadmat
import pandas as pd
import numpy as np


# Input: path (str)
# Output: data (Dataframe)
def data_loader(path):
    mat_data = loadmat(path)['data']
    index = np.arange(0, len(mat_data))
    df = pd.DataFrame(mat_data, index=index)
    return df.T
