import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn_genetic.callbacks import ProgressBar
from sklearn.linear_model import Lasso
from tqdm import tqdm


from boruta import BorutaPy
from sklearn_genetic import GAFeatureSelectionCV


def predict_model_loo(model, X, y, proba=False):
    y_pred = []
    y_true = []
    loo = LeaveOneOut()
    
    X = np.array(X)
    y = np.array(y)
    
    if X.ndim == 1:
        X = X[:, np.newaxis]
    
    for train_index, test_index in loo.split(X):
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

def forward_selection_get(model, X, y, columns=None, metric=balanced_accuracy, verbose=False):
    """
    Realiza una seleccion de caracteristicas hacia adelante para el 
    modelo, en el orden pasado en columns, utilizando la metrica seleccionada, 
    y realizando una validacion con Leave One Out (loo). Regresa las 
    caracteristicas seleccionadas, las puntuaciones de la metrica del modelo
    con las caracteristicas seleccionadas, y las puntuaciones de la metrica 
    del modelo en cada caracteristica.

    Parametros:
    ==========
    model: sklearn estimator
        Modelo que se utlizara al hacer la seleccion de caracteristicas. 
    X: pandas.DataFrame
        DataFrame con los datos para entrenar/validar el modelo, que contenga
        las caracteristicas como columnas y las filas como instancias.
    y: np.array o list (array-like)
        Target del modelo. 
    columns: np.array o list (array-like)
        El nombre de las columnas (caracteristicas) de donde se seleccionaran 
        las caracteristicas del modelo. 
    metric: function: y_true, y_pred -> score
        La metrica con la que se evaluara el modelo.
    verbose: bool (opcional)
        Si verbose es verdadero, imprime el numero de iteraciones restantes.

    Return:
    ===========
    features_selected: list
        Lista con las mejores caracteristicas encontradas.
    scores_model_selected: list
        La puntuacion de las caracteristicas seleccionadas al hacer la 
        seleccion hacia adelante.
    scores: list
        La puntuacion de cada caracteristica al ser agregada al modelo 
        junto con las anteriores.

    """
    if columns is None:
        columns = X.columns
        
    scores = []
    scores_model_selected = []
    features_selected = []
    
    n_cols = len(columns)
    
    for feature in tqdm(columns):
        features = features_selected + [feature]
        Xs = X[features]
        
        y_true, y_pred = predict_model_loo(model, Xs, y)
        score = metric(y_true, y_pred)
        scores.append(score)
        
        if not features_selected:
            features_selected.append(feature)
            scores_model_selected.append(score)
            continue
            
        if score > scores_model_selected[-1]:
            features_selected.append(feature)
            scores_model_selected.append(score)
        
        if verbose:
            print(n_cols, end='-')
            n_cols -= 1
        
    return features_selected, scores_model_selected, scores
    

data = pd.read_csv('data_preprocess/tran.csv')

columnas_target = ['Group', 'Positivo']
columnas_sintomas = ['Fever', 'Cough', 'Headache', 'Dyspnea', 'Diarrhea',
                     'Chest tightness', 'Chills', 'Pharyngalgia', 'Myalgia', 'Arthralgia',
                     'Arthralgia', 'Rhinorrhea', 'Polypnea', 'Anosmya', 'Dysgeusia']
columnas_clinicos = ['Age', 'Sex', 'Diabetes', 'Hipertension', 'Obesity', 'Smoking']
columnas_lab = list(
    set(data.columns)
    .difference(set(columnas_target + columnas_sintomas + columnas_clinicos))
)


###################     BORUTA     ###################
X = data[columnas_lab].values
y = data['Positivo'].values

print('BORUTA: ', end='')
model = RandomForestClassifier()
boruta = BorutaPy(model, n_estimators=500, max_iter=50, alpha=0.01)
boruta.fit(X, y)

columnas_boruta = []
for column, res_boruta in zip(columnas_lab, boruta.support_):
    if res_boruta:
        columnas_boruta.append(column)
        
with open('out/features_boruta.txt', 'w') as f:
    f.write('\n'.join(columnas_boruta))
    
print('DONE!!')


##########     SELECCION HACIA ADELANTE     ##########

X = data[columnas_lab]
y = data['Positivo']
features = list(pd.read_csv('out/univariable_rf.csv').feature)

print('\nFORWARD_SELECTION:')
model = RandomForestClassifier(n_estimators=500, max_depth=7)
columnas_forward, *_ = forward_selection_get(model, X, y, verbose=False, 
                                             columns=features, metric=f1_score)

with open('out/features_forward.txt', 'w') as f:
    f.write('\n'.join(columnas_forward))
    
print('DONE!!')


###########     ALGORITMOS GENETICOS     ############

X = data[columnas_lab]
y = data['Positivo']

print('\nALGORITMOS GENETICOS: ')
clf = RandomForestClassifier(n_estimators=100)
evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    cv=25,
    scoring='balanced_accuracy',
    population_size=50,
    generations=200,
    n_jobs=-1,
    verbose=False,
    keep_top_k=2,
    elitism=True,
    max_features=25,
    crossover_probability=0.7,
    mutation_probability=0.3,
)


evolved_estimator.fit(X, y, callbacks=ProgressBar())

columnas_ga = []

for support, col in zip(evolved_estimator.best_features_, columnas_lab):
    if support:
        columnas_ga.append(col)
        
with open('out/features_genetic.txt', 'w') as f:
    f.write('\n'.join(columnas_ga))
    
print('DONE!')


###################     LASSO     ####################

X = data[columnas_lab].values
y = data['Positivo'].values
features = columnas_lab

print('LASSO: ')
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', Lasso())
])

search = GridSearchCV(pipeline, 
                {'model__alpha': np.arange(0.1, 5, 0.1)}, cv=25, 
                scoring='neg_mean_squared_error', verbose=0
)

search.fit(X, y)


coefficients = search.best_estimator_.named_steps['model'].coef_
columnas_lasso = []
columnas_lasso_importance = []

for col, importance in zip(features, np.abs(coefficients)):
    if importance > 0:
        columnas_lasso.append(col)
        columnas_lasso_importance.append(importance)
        
with open('out/features_lasso.txt', 'w') as f:
    f.write('\n'.join(columnas_lasso))
    
print('DONE!!')