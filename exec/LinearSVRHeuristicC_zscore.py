import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

class LinearSVRHeuristicC_zscore(LinearSVR):
    """Inherit LinearSVR but overwrite fit function to set heuristically
    calculated C value in CV consistent manner without data leakage.
    """

    # inherit constructor entirely from LinearSVR

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):

        # z-score the features
        self.scaler_ = StandardScaler().fit(X)
        
        # calculate heuristic C
        X_trans = self.scaler_.transform(X)
        C = 1/np.mean(np.sqrt((X_trans**2).sum(axis=1)))

        # set C value
        self.C = C

        # call super fit method
        super().fit(X_trans, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn

    def predict(self, X=None):
        X_trans = self.scaler_.transform(X)
        return super().predict(X_trans)

    def score(self, X, y):
        X_trans = self.scaler_.transform(X)
        return super().score(X_trans, y)

