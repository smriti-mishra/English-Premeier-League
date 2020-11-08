# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 03:02:24 2018

@author: Versha Mom
"""

# 'data preprocessing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.svm import SVC
from pandas.tools.plotting import scatter_matrix
#import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
data = pd.read_csv('final_data.csv')
#scatter_matrix(data[['HTGD','ATGD','HTP','ATP','DiffFormPts']], figsize=(10,10))
from sklearn.ensemble import RandomForestClassifier
X_all = data.drop(['FTR'],1)
y_all = data['FTR']


from sklearn.preprocessing import scale
#Center to the mean and component wise scale to unit variance.
cols = [['HTGD','ATGD','HTP','ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])
#last 3 wins for both sides
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
#print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

from sklearn.cross_validation import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 760,
                                                    random_state = 2,
                                                    stratify = y_all)    

#for measuring training time
from time import time 

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    
    train_classifier(clf, X_train, y_train)
    
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

#clf_A = SVC(random_state = 912, kernel = 'linear')
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = GaussianNB()
clf_D = MultinomialNB()
clf_E = RandomForestClassifier(n_estimators=100)

#train_predict(clf_A, X_train, y_train, X_test, y_test)
#print('')
#train_predict(clf_B, X_train, y_train, X_test, y_test)
#print('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print('')
#train_predict(clf_D, X_train, y_train, X_test, y_test)
#print('')
train_predict(clf_E, X_train, y_train, X_test, y_test)  
print('')




#def logreg_train(x,alpha,y):
#    
#    y = y.replace(['H','NH'],[1,0])
#    r,colum = x.shape
#    b = np.zeros(colum+1)
#    x2 = x.values.tolist()
#    for j in range(r):
#        x1 = x2[j]
#        x1.insert(0,1)
#        output = np.sum([(b[i]*x1[i]) for i in range(colum+1)])
#        predict = 1 / (1 + np.exp(-output))
#        b = [(b[i]+ alpha * (y[j] - predict) * predict * (1-predict) * x1[i]) for i in range(colum+1)]
#    return b
#

#def logreg_predict(x,y):
#    
#    y = y.replace(['H','NH'],[1,0])
#    r,colum = x.shape
#    b = logreg_train(X_train,0.4,y_train)
#    start = time()
#    x2 = x.values.tolist()
#    predict = np.zeros(r)
#    for j in range(r):
#        x1 = x2[j]
#        x1.insert(0,1)
#        output = np.sum([(b[i]*x1[i]) for i in range(colum+1)])
#        val = 1 / (1 + np.exp(-output))
#        if val  <= 0.5:
#            predict[j] = 0
#        else:
#            predict[j] = 1
#            
#    for i in range(r):
#        correct_predict = 0
#        if (y[i]==predict[i]):
#            correct_predict = correct_predict + 1
#        
#    accuracy = (correct_predict/r)*100
#    end = time()
#    print("Trained model in {:.4f} seconds".format(end - start))            
#    print("accuracy score for training set: {:.4f} .".format(accuracy))
    
#logreg_predict(X_test,y_test)

def baseline(y):
    ''' Prediction based on Zero Rule Prediction '''
    
    count = 0
    for i in range(len(y)):
        if y[i] == 'H':
            count+= 1
        
    acc = (count/len(y))*100
    print('The accuracy of baseline {}:'.format(acc))

    
baseline(y_all)