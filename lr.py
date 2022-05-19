from sklearn.linear_model import LinearRegression #線性迴歸
import matplotlib.pyplot as plt
from sklearn import datasets #測試資料庫

x,y = datasets.make_regression(n_samples = 200, n_features =1, n_targets =1, noise = 17)
model = LinearRegression()
model.fit(x,y)
predict = model.predict(x[:200,:])
plt.plot(x,predict,c="red")
plt.scatter(x,y,linewidths=0.1)
plt.show()