#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
### KNN
### using KNN(K=5) to classify the test data and gain accuracy=0.920 using 0.009sec 
### using KNN(K=7) to classify the test data and gain accuracy=0.936 using 0.008sec
### using KNN(K=9) to classify the test data and gain accuracy=0.936 using 0.008sec
### using KNN(K=7), p=1 to classify the test data and gain accuracy=0.932 using 0.009sec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import datetime
print datetime.datetime.now()
clf = KNeighborsClassifier(n_neighbors=7, algorithm='brute', metric='euclidean')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
print acc
print datetime.datetime.now()
### SVM
### using rbf to classify the test data and gain accuracy=0.92 using 0.008sec
### using rbf and C=100000000 to classify the test data and gain accuracy=0.956 using 0.074sec
from sklearn.svm import SVC
print datetime.datetime.now()
clf_svm = SVC(kernel='rbf', C=100000000)
clf_svm.fit(features_train, labels_train)
pred = clf_svm.predict(features_test)
acc = accuracy_score(labels_test, pred)
print acc
print datetime.datetime.now()
### tree
### using entropy and min 40 to classify the test data and gain accuracy=0.912 using 0.007sec
from sklearn import tree
print datetime.datetime.now()
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=40)
clf_tree.fit(features_train, labels_train)
pred = clf_tree.predict(features_test)
acc = accuracy_score(labels_test, pred)
print acc
print datetime.datetime.now()

### RandomForestClassifier
### using estimators=10 to classify the test data and gain accuracy=0.924 using 0.633sec
from sklearn.ensemble import RandomForestClassifier
print datetime.datetime.now()
clf_rfc = RandomForestClassifier(n_estimators=10)
clf_rfc.fit(features_train, labels_train)
pred = clf_rfc.predict(features_test)
acc = accuracy_score(labels_test, pred)
print acc
print datetime.datetime.now()





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
