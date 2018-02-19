import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error as mse
from statistics import mean

from sklearn.model_selection import cross_val_score
# utility self library
from utils import *
from sklearn.model_selection import train_test_split


def GBR():
    '''
    Create and evaluate the GradientBoostingRegressor
    '''
    data, labels = load_data()
    # params
    params = { 'min_samples_split': 2, 'loss': 'lad', }
    # model
    clf = ensemble.GradientBoostingRegressor(**params)
    # cross validation
    scores = cross_val_score(clf, data, labels, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    print(scores)
    print('-------------------------------------------------------------')
    print('MSE:', str(scores.mean()),  str(scores.std()) )



if __name__ == '__main__':
    GBR()
