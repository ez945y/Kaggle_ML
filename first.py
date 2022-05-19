from sklearn import tree #二元樹分類
features = [[150,1],[170,1],[130,0],[140,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
wantPredict = clf.predict([[160,1]]) 
if wantPredict == [1]:
    print("這是蘋果")
elif wantPredict == [0]:
    print("這是橘子")
