from sklearn.model_selection import LeaveOneOut
import numpy as np
from tqdm import tqdm

def predict_model_loo(model, X, y, proba=False):
    y_pred = []
    y_true = []
    loo = LeaveOneOut()
    
    X = np.array(X)
    y = np.array(y)
    
    if X.ndim == 1:
        X = X[:, np.newaxis]
    
    for train_index, test_index in tqdm(loo.split(X)):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        if not proba:
            yi_pred = model.predict(X_test)[0]
        else:
            yi_pred = model.predict_proba(X_test)[0, 1]
        
        y_true.append(y_test[0])
        y_pred.append(yi_pred)
            
    return y_true, y_pred
