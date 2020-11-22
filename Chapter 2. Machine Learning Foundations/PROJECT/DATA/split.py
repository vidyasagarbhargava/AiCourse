import numpy as np
dont run this file
import pandas as pd

d = pd.read_csv('winequality-red.csv', delimiter=';')
print(len(d))

train_idxs = np.random.choice(range(len(d)), size=int(0.8*len(d)))
train = d.iloc[train_idxs]
print(len(train))

test_idxs = np.full(len(d), True)
test_idxs[train_idxs] = False
test = d.iloc[test_idxs]
print(len(test))

train.to_csv('winequality-red-train.csv')
test.to_csv('winequality-red-test.csv')