from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

lb_model = preprocessing.LabelBinarizer()
lb_model.fit([1, 2, 6, 4, 2])

print(lb_model.classes_)
transform_label = lb_model.transform([1,6,3])
print(transform_label)

lb_model.fit(['yes','no','no','yes'])
print(lb_model.classes_)


print(lb_model.fit_transform(['yes','no','no','yes']))


import numpy as np

lb_model.fit(np.array([[0,0,0],[1,0,0]]))
print(lb_model.classes_)
print(lb_model.transform([0, 1, 2, 1]))