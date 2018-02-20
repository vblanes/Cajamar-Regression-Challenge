import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.model_selection import cross_val_score

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



if __name__ == '__main__':

    data = pd.read_csv('TRAIN.txt', sep=",")
    #data.columns.values

    labels = data['Poder_Adquisitivo']

    labelsInf1M = labels[ labels < 1000000]
    labelsInf2KK = labels[ labels < 200000 ]
    labelsInf1KK = labels[ labels < 100000 ]
    labelsInf60K = labels[ labels < 60000 ]
    media = labels.mean()
    mediana = labels.median()

    print(str(labels.count()))
    print(str(labelsInf1M.count()))
    print(str(labelsInf2KK.count()))
    print(str(labelsInf1KK.count()))
    print(str(labelsInf60K.count()))
    print(str(media))
    print(str(mediana))

    n, bins, patches = plt.hist(labelsInf60K, bins=100)
    plt.title('Histograma Poder_Adquisitivo')
    plt.show()
