import numpy as np
from sklearn.base import BaseEstimator

class GroupsImputer(BaseEstimator):
    def __init__(self, method='mode', groups=None):
        self.method='mode'
        self.groups=groups
        
    def fit(self, X, y=None):
        if y is None:
            y = self.groups
            
        fillers_column = {}
        
        for col in X.columns:
            fill_column_values = []
            column = X[col]
            for group in [1, 2, 3, 4]:
                data = column[y==group]
                fill_value = data.mode()[0]
                fill_column_values.append(fill_value)
            fillers_column[col] = fill_column_values
            
        self.fillers_column = fillers_column
        return self
        
    def transform(self, X, y):
        X_filled = X.copy()
        for col in X.columns:
            for i, index in enumerate(X_filled[col].index):
                if np.isnan(X.loc[index, col]):
                    group = y[index]
                    fill_value = self.fillers_column[col][group-1]
                    X_filled.loc[index, col] = fill_value
     
        return X_filled