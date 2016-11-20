#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV




### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# use all dataset and linear kernel/C=default to train classify
# Accuracy is 0.984072810011
'''clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc
report = classification_report(pred, labels_test)
print report'''

'''
# use 1% dataset and linear kernel/C=default to train classify
# Accuracy is 0.884527872582
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc
report = classification_report(pred, labels_test)
print report
# use 1% dataset and linear kernel/C=10000 to train classifier
# Accuracy is 0.8606370876
clf = SVC(kernel='linear', C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc
report = classification_report(pred, labels_test)
print report

# use 1% dataset and rbf kernel/C=10000 to train classifier
# Accuracy is 0.892491467577
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc

# use 1% dataset and kernel=gamma, C=1000, gamma=0.001 to train classifier
# parameter is search by GridSearchCV
# Accuracy is 0.896473265074
parameters = {'kernel':('rbf','linear'), 'C':[1e3, 5e3, 1e4, 5e4, 1e5, 1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc'''

# use x% dataset and kernel=gamma, C=1000, gamma=0.001 to train classifier
# parameter is search by GridSearchCV
# Accuracy is 
# /60 => 0.9
# /50(2%) => 0.922639362912
# /40(2.5%) => 0.930602957907
# /10(10%) => 0.963026166098
# /8(12.5%) => 0.960750853242
# /6(17%) => 0.970989761092
# /5(20%) => 0.978384527873
# /4(25%) => 0.976109215017
# /3(33%) => 0.9795221843
# /2(50%) => 0.98236632537
# We can use 10% ~ 50% of data to train a classfier
# And the clasifier is as good as linear kernel with 100% dataset
features_train = features_train[:len(features_train)/2] 
labels_train = labels_train[:len(labels_train)/2]

clf = SVC(kernel='rbf', C=10000, gamma=0.001)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc
#########################################################



