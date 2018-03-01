from sklearn import datasets
iris = datasets.load_iris()
print(iris.data)
print(len(iris.data))
print(iris.target)
digits = datasets.load_digits()

from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100)

clf.fit(digits.data[:-1],digits.target[:-1])
predict_target = clf.predict(digits.data[-1:])
print(predict_target)

import matplotlib.pyplot as plt

images_and_labels_test_datasets = list(zip(digits.images,digits.target))