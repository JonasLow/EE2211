'''
Question 4:
(a) Those features with very large values may overshadow those with very small values
(b) We can either use min-max or z-score normalisation to resolve the problem
'''

import pandas as pd
import numpy as np

# part(a)
dataset = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None)
print(dataset.describe())

print()
# part (b)
print((dataset[[1, 2, 3, 4, 5]] == 0).sum())

print()
# part (c)
dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, np.NaN)
print(dataset.head(20))
print()
print(dataset.isnull().sum())

'''
Question 6:
Data is not always unbiased (False)

Question 7:
DORSCON is Categorical and Ordinal (1, 2)

Question 8:
Median, 50%
'''