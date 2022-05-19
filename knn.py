from sklearn import datasets #測試資料庫
from sklearn.model_selection import train_test_split #切割
from sklearn.neighbors import KNeighborsClassifier #K-近鄰演算法
import numpy as np

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target
train_data, test_data, train_label, test_label = train_test_split(iris_data,iris_label,test_size = 0.2)
knn = KNeighborsClassifier()
knn.fit(train_data, train_label)
print(knn.predict(test_data))
print(test_label)