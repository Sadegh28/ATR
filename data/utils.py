from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics

import numpy as np
import os

def read_data(d_name, d_path = None): 
    if d_path == None: 
        mulan_datasets = [x[0] for x in available_data_sets().keys()]
        if d_name not in mulan_datasets:
            raise ValueError('{} not found in our database'.format(d_name)) 
        X_train, y_train, _, _ = load_dataset(d_name, 'train')
        X_test, y_test, _, _ = load_dataset(d_name, 'test')
        return X_train.toarray(), y_train.toarray(), X_test.toarray(), y_test.toarray()

    if not (os.path.isdir(d_path)):
        raise ValueError('data directory {} not found'.format(d_path))
    X_train_file = d_path  + d_name + '\\' + 'train' + '.csv'
    y_train_file = d_path  +d_name + '\\'+ 'train_labels' + '.csv'
    X_test_file = d_path  +d_name + '\\'+ 'test' + '.csv'
    y_test_file = d_path  +d_name + '\\'+ 'test_labels' + '.csv'

    
    X_train, y_train =  np.genfromtxt(X_train_file, delimiter=','),  np.genfromtxt(y_train_file, delimiter=',')
    X_test, y_test =  np.genfromtxt(X_test_file, delimiter=','),  np.genfromtxt(y_test_file, delimiter=',')
    return X_train[1:,1:], y_train[1:,1:], X_test[1:,1:], y_test[1:,1:]

def classify(X_train, y_train, X_test, y_test, classifier, mtrs): 
    results = {}
    if classifier =='MLKNN':
        clf = MLkNN(k=10)
    if classifier == 'BinaryRelevance':
        clf  = BinaryRelevance( classifier = MultinomialNB())
    
    
    prediction = clf.fit(X_train, y_train).predict(X_test)
    for m in mtrs: 
        if m == 'hamming loss': 
            results[m] = metrics.hamming_loss(y_test, prediction.toarray())
            
        if m == 'label ranking loss': 
            results[m] = metrics.label_ranking_loss(y_test, prediction.toarray())

        if m == 'coverage error':
            results[m] = metrics.coverage_error(y_test, prediction.toarray())

        if m == 'average precision score':
            results[m] = metrics.average_precision_score(y_test, prediction.toarray())
        
    return results
                


