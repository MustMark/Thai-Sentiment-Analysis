import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

class LexiconTransformer(BaseEstimator, TransformerMixin):

    positive_words = ["ดี", "เยี่ยม", "สุดยอด", "คุ้ม", "อร่อย", "ชอบ", "ส่งเสริม", "สุข"]
    negative_words = ["แย่", "ห่วย", "พัง", "แพง", "ช้า", "ไม่ดี"]

    def lexicon_features(self, text):
        pos = sum(word in text for word in self.positive_words)
        neg = sum(word in text for word in self.negative_words)
        return [pos, neg]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = np.array([self.lexicon_features(text) for text in X])
        return csr_matrix(feats)