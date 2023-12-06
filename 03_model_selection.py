import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation import *
import metrics
from sklearn.metrics import roc_auc_score as auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings

# Archivos con los metabolitos seleccionados
files_features_selected = {
    'boruta': 'out/features_boruta.txt',
    'forward': 'out/features_forward.txt', 
    'genetic': 'out/features_genetic.txt',
    'lasso': 'out/features_lasso.txt'}


# Modelos de ML y Cuadricula de busqueda
models = [
    SVC, 
    RandomForestClassifier, 
    GaussianNB, 
    LogisticRegression
]


SVC_grid_sarch = {
    'kernel': ['linear',  'sigmoid'],
    'C': [1, 10, 100, 1_000, 10_000, 1_000_000],
    'max_iter': [500_000_00],
    'probability': [True]
}

RF_grid_sarch = {
    'n_estimators': [100, 250, 500, 750],
    'max_depth': [5, 7, 11, None],
    'bootstrap': [False, True],
}

GaussNB_grid_search = {
}

Logr_grid_search = {
    'C': [1, 10, 100, 1000, 1_0000, 100_000, 1_000_000],
    'max_iter': [500_000_000],
    'fit_intercept': [True, False]
}

search_grids = {
    SVC.__name__: SVC_grid_sarch,
    RandomForestClassifier.__name__: RF_grid_sarch,
    GaussianNB.__name__: GaussNB_grid_search,
    LogisticRegression.__name__: Logr_grid_search
}



data = pd.read_csv('./data_preprocess/tran.csv')

# Raliza los modelos y hace las predicciones LOO de cada uno
predictions = {}
best_score = 0
best_model = {}
scoring_metric = metrics.balanced_accuracy

for selection_method in files_features_selected:
    with open(files_features_selected[selection_method]) as f:
        features = f.readlines()
        features = [feature.strip() for feature in features]
        
    X = data[features]
    y = data['Positivo']
    
    prediction_models = {}
    for model_obj in models:
        # Busqueda
        search_grid = search_grids[model_obj.__name__]
        Search = GridSearchCV(model_obj(), search_grid, scoring='balanced_accuracy',
                              cv=10, return_train_score=True, verbose=True)
        Search.fit(X, y)
        best_model_params = Search.best_params_
        
        model = model_obj(**best_model_params)
        ytrue, ypred = predict_model_loo(model, X, y, proba=True)
        prediction_models[model_obj.__name__] = {'true':ytrue, 'ypred':ypred}
        
        # Scoring and save best model params
        model_score = scoring_metric(ytrue, ypred)
        if model_score > best_score:
            best_score = model_score
            best_model_params = Search.best_params_
            
    predictions[selection_method] = prediction_models
    

# Metricas de evaluacion de los modelos
balanced_accuracies = {}
f1_scores = {}
roc_scores = {}
precisions = {}
recalls = {}
specificities = {}
aucs = {}

for selection_method in predictions:
    balanced_accuracies[selection_method] = {}
    f1_scores[selection_method] = {}
    precisions[selection_method] = {}
    recalls[selection_method] = {}
    specificities[selection_method] = {}
    aucs[selection_method] = {}
    
    for model in models:
        ytrue, ypred = predictions[selection_method][model.__name__].values()
        model_balanced_accuracy = metrics.balanced_accuracy(ytrue, ypred)
        model_f1_score = metrics.f1_score(ytrue, ypred)
        model_precision = metrics.precision(ytrue, ypred)
        model_recall = metrics.recall(ytrue, ypred)
        model_specificity = metrics.specificity(ytrue, ypred)
        model_auc = auc(ytrue, ypred)

        balanced_accuracies[selection_method][model.__name__] = model_balanced_accuracy
        f1_scores[selection_method][model.__name__] = model_f1_score
        precisions[selection_method][model.__name__] = model_precision
        recalls[selection_method][model.__name__] = model_recall
        specificities[selection_method][model.__name__] = model_specificity
        aucs[selection_method][model.__name__] = model_auc
        
# Genera un DataFrame de los modelos para cada metrica y las guarda en pdf's.
df_balanced_accuracies = pd.DataFrame(balanced_accuracies)
df_f1_scores = pd.DataFrame(f1_scores)
df_auc_scores = pd.DataFrame(aucs)
df_precisions = pd.DataFrame(precisions)
df_recalls = pd.DataFrame(recalls)
df_specificities = pd.DataFrame(specificities)

df_balanced_accuracies.to_csv('out/metrics_model_selection/balanced_accuracies.csv')
df_f1_scores.to_csv('out/metrics_model_selection/f1_scores.csv')
df_auc_scores.to_csv('out/metrics_model_selection/auc_scores.csv')
df_precisions.to_csv('out/metrics_model_selection/precisions.csv')
df_recalls.to_csv('out/metrics_model_selection/recalls.csv')
df_specificities.to_csv('out/metrics_model_selection/specificities.csv')

with open('out/best_model/best_model.pkl', 'wb') as fp:
    pickle.dump(best_model, fp)
# Guarda un DataFrame del mejor modelo.