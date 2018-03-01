import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=5)
plt.scatter(X,y)
plt.show()