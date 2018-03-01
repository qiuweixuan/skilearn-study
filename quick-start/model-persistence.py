from sklearn import  svm,datasets

iris = datasets.load_iris()
n_samples = len(iris.data)
train_size = n_samples * 2 // 3
train_X,train_y = iris.data[:train_size],iris.target[:train_size]

classifier = svm.SVC()
classifier.fit(train_X,train_y)

import pickle

model_file = open('irisModel.txt','wb')
pickle.dump(classifier,model_file)
model_file.close()

model_file = open('irisModel.txt','rb')
used_clf = pickle.load(model_file)
model_file.close()

predict_y = used_clf.predict(iris.data[train_size : train_size + 1])
print(predict_y[0])

