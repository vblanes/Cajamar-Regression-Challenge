'''
Utility functions for data handeling
'''
import pandas as pd
import numpy as np

def factorization(column):
    aux_ = dict()
    new_col = []
    index = 1
    for el in column:
        if el in aux_:
            new_col.append(aux_[el])
        else:
            aux_[el] = index
            new_col.append(index)
            index += 1
    return new_col


def load_data():
    data = pd.read_csv('TRAIN.txt', sep=",")
    print(data.shape)
    del data['ID_Customer']
    #data.columns.values

    labels = data['Poder_Adquisitivo']
    del data['Poder_Adquisitivo']

    data['Socio_Demo_01'] = factorization(data['Socio_Demo_01'])
    return data, labels
