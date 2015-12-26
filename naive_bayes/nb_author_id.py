#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB

t0 = time()
print 'starting training'

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"
### training time: 6.393 s

t1 = time()
print 'starting prediction'

pred = clf.predict(features_test)

print "prediction time:", round(time()-t1, 3), "s"
### prediction time: 0.188 s

t2 = time()
print 'starting scoring'

index = 0
matches = 0
for value in pred:
    test_value = labels_test[index]
    if value == test_value:
        matches += 1
    index += 1

total = len(pred);

print (matches)
print (total)
### calculate and return the accuracy on the test data
### this is slightly different than the example,
### where we just print the accuracy
### you might need to import an sklearn module
accuracy = 1.0 * matches /  total
print accuracy
### 0.973265073948

### http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

from sklearn.metrics import accuracy_score

score = accuracy_score(labels_test, pred)
print score
### 0.973265073948
print "scoring time:", round(time()-t2, 3), "s"








#########################################################
### your code goes here ###


#########################################################


