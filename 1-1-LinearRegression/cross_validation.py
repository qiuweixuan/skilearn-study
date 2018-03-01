from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

iris_data = datasets.load_iris()

data_X = iris_data.data
data_y =iris_data.target
knn_model = KNeighborsClassifier()
scores = cross_val_score(knn_model,data_X,data_y,cv=5,scoring='accuracy')
print(scores)
print(scores.mean())

k_range = range(1,31)
k_score = []
for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, data_X, data_y, cv=5, scoring='accuracy')
    #k_score.append(scores)
    k_score.append(scores.mean())

plt.plot(k_range,k_score)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()