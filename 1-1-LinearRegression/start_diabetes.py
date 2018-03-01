import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

boston_data = datasets.load_boston()

data_X = boston_data.data
data_y = boston_data.target

X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,test_size = 0.3)
#print(len(X_train),len(y_train))

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

predicted = lr_model.predict(X_test)
print(predicted)
print(y_test)

print(lr_model.coef_)
print(lr_model.intercept_)

print(lr_model.get_params())

print(lr_model.score(X_test,y_test))