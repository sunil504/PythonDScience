from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,52,38]]
Y = ['male','female','female','female','male','male','male','male','female']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
predection = clf.predict([[190,70,43],[190,70,43]])
print('tree')
print(predection)

#LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,Y)
Lpred = logreg.predict([[181,80,44],[190,70,43]])
print(Lpred)

# RandomForest 

RF = RandomForestClassifier(n_estimators = 10)
RF = RF.fit(X,Y)
RFpred = RF.predict([[181,80,44]])
print(RFpred)
