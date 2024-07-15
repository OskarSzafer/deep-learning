import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

import deep_learning as dl

nn = dl.NN(2, 3, 2, 2)
nn.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0, 1], [1, 0], [1, 0], [0, 1]]), epochs = 1000)
out = nn.run(np.array([1, 0]))
print(out)