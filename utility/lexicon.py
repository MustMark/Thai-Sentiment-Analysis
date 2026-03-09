import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from utility.preprocess import lexicon_features

class LexiconTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = np.array([lexicon_features(text) for text in X])
        return csr_matrix(feats)