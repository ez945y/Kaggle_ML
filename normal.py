from sklearn import preprocessing #標準化
from sklearn.model_selection import train_test_split #切分
from sklearn.datasets import make_blobs #產生資料
from sklearn.svm import SVC #SVC分類法
import numpy as np #矩陣
import matplotlib.pyplot as plt #繪圖
x, y = make_blobs(n_samples=300,cluster_std=1.0)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()




x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
clf = SVC()
clf.fit(x_train,y_train)
note = clf.score(x_test,y_test)
print(note)