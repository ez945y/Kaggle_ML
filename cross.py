from sklearn.model_selection import cross_val_score,train_test_split #交叉驗證, 切割
from sklearn import datasets #測試資料庫
from sklearn.neighbors import KNeighborsClassifier #K-近鄰演算法
import numpy as np #矩陣
import matplotlib.pyplot as plt #繪圖
iris = datasets.load_iris()
X = iris.data
y = iris.target
k_range = range(1,31)
k_scores = []
for k_number in k_range:
    knn = KNeighborsClassifier(n_neighbors=k_number)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()