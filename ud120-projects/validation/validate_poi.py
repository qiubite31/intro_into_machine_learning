#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
from sklearn.tree import tree 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.cross_validation import train_test_split

# using the default data to train will get the higher training accuracy 0.989473684211
# that's overfit
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(labels, pred)
print acc
# split dataset into train and test dataset
# using test dataset to test the traing model will get the lower accuracy 0.724137931034
# split traing/test dataset will prevent model overfit
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
print len(X_test)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print acc
print "Unit 14-28: How many POIs are in the test set for your POI identifier?"

print sum(pred)

print "How many people are in the test set?"
print len(pred)

print "If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?"
print accuracy_score(y_test, [0]*len(y_test))

print "Look at the predictions of your model and compare them to the true test labels."
print "Do you get any true positives?"

print confusion_matrix(y_test, pred)
print classification_report(y_test, pred)

print "precision score:"
print precision_score(y_test, pred)

print "recall score:"
print recall_score(y_test, pred)

print "hypothetical test set:"
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print confusion_matrix(true_labels, predictions)

print "hypothetical precision score:"
print precision_score(true_labels, predictions)

print "hypothetical recall score:"
print recall_score(true_labels, predictions)
