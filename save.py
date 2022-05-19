from sklearn import svm #SVM 演算法
from sklearn import datasets #測試資料庫
import joblib

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data , iris.target
clf.fit(X,y)

joblib.dump(clf,'clf.pkl')