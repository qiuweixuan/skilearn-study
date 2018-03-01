import numpy as np
from  sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size = 0.3)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)

predicted = knn_model.predict(X_test)
print(predicted)
print(y_test)