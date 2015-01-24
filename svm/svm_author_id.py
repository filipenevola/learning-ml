#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""

import sys
import time

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=10000.0)
t0 = time.time()
clf.fit(features_train, labels_train)
print "training time:", round(time.time() - t0, 3), "s"
#training time: 178.676 s


t0 = time.time()
pred = clf.predict(features_test)
print "predict time:", round(time.time() - t0, 3), "s"
#predict time: 18.525 s

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, pred))
#0.984072810011

print(accuracy_score(labels_test, pred, normalize=False))
#1730

print(pred[10])
print(pred[26])
print(pred[50])

cont = 0
for p in pred:
    if p == 1:
        cont += 1

print cont
