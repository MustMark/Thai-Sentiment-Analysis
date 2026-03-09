import numpy as np

def nb_features(X, y):
    
    y = np.array(y)

    pos = X[y == 1].sum(axis=0) + 1
    neg = X[y == 0].sum(axis=0) + 1

    r = np.log(pos / neg)

    return np.asarray(r).ravel()