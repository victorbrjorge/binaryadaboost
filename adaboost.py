#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import csv
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def get_tictactoe_dataset ():
    with open('tic-tac-toe.data') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
            if row[-1] == 'positive':
                row[-1] = int(1)
            else:
                row[-1] = int(-1)
            data.append(row)
    csvfile.close()
    return np.array(data)

'''
    # receives the data matrix, the chosen feature and value to build a decision stump
    # inverse decide if the correct answer is 1 or -1
    # returns an array of {-1, 1} with the predicted class for each example of the data matrix
'''
def decision_stump(data, feature, value, inverse): #implementation of a generic decision stump
    
    resp = np.ones(data.shape[0])
    if not inverse:
        resp[data[:,feature] != value] = -1.0 
    else:
        resp[data[:,feature] == value] = -1.0
    return resp


'''
    # test all possible stumps and calculate the error for each one considering examples weights
    # returns the stump with the least error, the prediction made by that stump and its error
'''
def get_best_stump(data, labels, weights):
    
    m,n = data.shape
    best_stump = {}
    best_predict = np.zeros(m)
    
    min_err = float('inf')
    
    for i in range(n-1):
        for value in ['x', 'o', 'b']:
            for inverse in [True, False]: #loop through the 54 possible stumps to select the one with the least error
                predict = decision_stump(data, i, value, inverse)
                
                err = np.ones(m, dtype=np.int)
                err[predict == labels] = 0
                weighted_err = np.dot(weights, err)

                #print i, value, inverse, weighted_err
                
                if weighted_err < min_err:
                    min_err = weighted_err

                    best_predict = predict.copy()
                    best_stump['feature'] = i
                    best_stump['value'] = value
                    best_stump['inverse'] = inverse
    
    return best_stump, best_predict, min_err


def adaboost(data, num_iter=1000):
    weak_models = []
    error_per_it = []
    stump_error = []
    m = data.shape[0]
    weights = np.array([1.0/m for x in range(m)])
    labels = np.array(data[:,-1], dtype=np.float)
    ensemble_predict = np.zeros(m)
    
    for i in range(num_iter):
        stump, predict, error = get_best_stump(data, labels, weights)
        

        alpha = float(0.5*np.log((1.0-error) / (error+1e-15)))
        stump['alpha'] = alpha
        weak_models.append(stump)

        weights = np.multiply(weights, np.exp(np.multiply(-alpha*labels, predict)))
        weights = weights/weights.sum()

        ensemble_predict += predict * alpha
        ensemble_error = np.multiply(np.sign(ensemble_predict) != labels, np.ones(m))
        ensemble_error = ensemble_error.sum()/m
        
        error_per_it.append(ensemble_error)
        stump_error.append(error)
    return weak_models, error_per_it, stump_error

def predict(example, model):
    ensemble_predict = 0
    for i in range(len(model)):
        predict = stump_predict(example, model[i]['feature'],model[i]['value'],model[i]['inverse'])
        ensemble_predict += model[i]['alpha'] * predict

    return np.sign(ensemble_predict)

def stump_predict(example, feature, value, inverse): #implementation of a generic decision stump    
    resp = 1.0
    if not inverse and example[feature] != value:
        resp = -1.0 
    elif inverse and example[feature] == value:
        resp = -1.0
    return resp

def test_model(data, model):
    
    m = data.shape[0]  
    labels = np.array(data[:,-1], dtype=np.float)
    predicted_labels = np.zeros(m)
    error = np.zeros(m)
    
    for i in range(m):
        predicted_labels[i] = predict(data[i,:], model)
    
    error = np.multiply(np.sign(predicted_labels) != labels, np.ones(m))
    return error.sum()/m

if __name__ == '__main__':
    data = get_tictactoe_dataset()
    
    kf = KFold(n_splits=5, shuffle=True)
    
    test_error = []
    
    for i in range(50,301,50):
        print i
        if i == 300:
            train_error = np.zeros(i)
            stump_error = np.zeros(i)
            error = 0
            for train, test in kf.split(data):
                model, err, st_err = adaboost(data[train], i)
        
                train_error = np.add(train_error, err)
                stump_error = np.add(stump_error, st_err)

                error += test_model(data[test], model)

            test_error.append(error/5)
            train_error = train_error/5
            stump_error = stump_error/5
        else:
            error = 0
            for train, test in kf.split(data):
                model, err, st_err = adaboost(data[train], i)
                error += test_model(data[test], model)
                
            test_error.append(error/5)
    
    output_1 = open('train_error', 'w')
    output_2 = open('stump_error', 'w')
    output_3 = open('test_error', 'w')
    np.savetxt(output_1, train_error)
    np.savetxt(output_2, stump_error)
    np.savetxt(output_3, test_error)