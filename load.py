from sklearn import svm #SVM 演算法
from sklearn import datasets #測試資料庫
import joblib

iris = datasets.load_iris()
X,y = iris.data , iris.target
clf2 = joblib.load('clf.pkl')
print(X[0:1])
print()
print(X[1:2])
print()
for i in range(0,2):
    print(X[i:i+1])