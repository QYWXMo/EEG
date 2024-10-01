import numpy as np


list=[1,2,3,4,5,6,7,8,9]
arr=np.array(list)
print(arr.shift(periods=1))
print(arr.shift(periods=2))
print(arr[:-1])