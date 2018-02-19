import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error as mse
from statistics import mean

from sklearn.model_selection import cross_val_score
# utility self library
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
#
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# para fully-conected
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import GaussianNoise as GN
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def GBR():
    '''
    Create and evaluate the GradientBoostingRegressor
    '''
    data, labels = load_data()
    # params
    params = { 'min_samples_split': 2, 'loss': 'ls', }
    # model
    clf = ensemble.GradientBoostingRegressor(**params)
    # cross validation
    scores = cross_val_score(clf, data, labels, cv=10, n_jobs=-1, scoring=make_scorer(score_func=mse, greater_is_better=True))
    print(scores)
    print('-------------------------------------------------------------')
    print('MSE:', str(scores.mean()),  str(scores.std()) )


def model_keras():
    # crea el modeloTrue
    model = Sequential()
    # primera capa
    model.add(Dense(128, input_shape=(87,)))
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    # segunda capa
    model.add(Dense(256))
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    # tercera capa
    model.add(Dense(256))
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    # capa de salida
    model.add(Dense(1))
    model.add(GN(0.3))
    model.add(Activation('relu'))

    # compilar el modelo
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

def GBR_60k():
    '''
    Create and evaluate the GradientBoostingRegressor
    '''
    data, labels = load_data()
    # delete the register above 60k
    selecter = [pa <= 60000 for pa in labels]
    data = data[selecter]
    labels = [pa for ind, pa in enumerate(labels) if selecter[ind]]

    # params
    params = { 'min_samples_split': 2, 'loss': 'ls', }
    # model
    clf = ensemble.GradientBoostingRegressor(**params)
    # cross validation
    scores = cross_val_score(clf, data, labels, cv=10, n_jobs=-1, scoring=make_scorer(score_func=mse, greater_is_better=True))
    print(scores)
    print('-------------------------------------------------------------')
    print('MSE:', str(scores.mean()),  str(scores.std()) )


if __name__ == '__main__':
    '''
    # seed
    seed = np.random.seed(42)
    # pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=model_keras, epochs=2, batch_size=50, verbose=0)))
    pipeline = Pipeline(estimators)
    # load data
    X, y = load_data()
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, y, cv=10, n_jobs=-1, scoring=make_scorer(score_func=mse, greater_is_better=True))
    print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    '''
    GBR_60k()
