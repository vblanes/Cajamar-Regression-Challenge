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
    '''
    scores = cross_val_score(GBR, data, labels, cv=10, n_jobs=-1)
    print(scores)
    print('-------------------------------------------------------------')
    print('Desviacion absoluta:', str(scores.mean()),  str(scores.std()) )
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return mse(y_test, predicted), clf.feature_importances_


if __name__ == '__main__':
    mses = []
    imps = []
    for _ in range(10):
        m, v = GBR()
        mses.append(m)
        imps.append(v)
    print('mse', str(mean(mses)))
    data, _ = load_data()
    data = data.columns.values
    imps = sum(imps)/len(mses)
    for i, cl in enumerate(data):
        print(cl, ';', imps[i])
