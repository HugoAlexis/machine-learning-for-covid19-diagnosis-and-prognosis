import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import metrics
from metrics import *
from evaluation import predict_model_loo

train = pd.read_csv('data_preprocess/tran.csv')
Xtrain = train.drop(columns='Positivo')
ytrain = train['Positivo']

##################################
columnas_target = ['Group', 'Positivo']
columnas_sintomas = ['Fever', 'Cough', 'Headache', 'Dyspnea', 'Diarrhea',
                     'Chest tightness', 'Chills', 'Pharyngalgia', 'Myalgia', 'Arthralgia',
                     'Arthralgia', 'Rhinorrhea', 'Polypnea', 'Anosmya', 'Dysgeusia']
columnas_clinicos = ['Age', 'Sex', 'Diabetes', 'Hipertension', 'Obesity', 'Smoking']
columnas_lab = list(
    set(train.columns)
    .difference(set(columnas_target + columnas_sintomas + columnas_clinicos))
)

obj_models = [SVC, RandomForestClassifier, LogisticRegression, GaussianNB]
models_uni = {}

for col in columnas_lab:
    X = train[[col]]
    col_scores = {}
    for obj_model in obj_models:
        model = obj_model()
        ytrue, ypred = predict_model_loo(model, X, ytrain)
        
        r = select_threshold(ytrue, ypred)
        acuracy_score = balanced_accuracy(ytrue, ypred, r=r)
        col_scores[obj_model.__name__] = acuracy_score
        
    models_uni[col] = col_scores
    
    
df_models_uni = pd.DataFrame(models_uni).T

df_models_uni = df_models_uni.assign(
                    best_model = df_models_uni.idxmax(axis=1),
                    best_score = df_models_uni.max(axis=1)
).sort_values(by='best_score', ascending=False)

df_models_uni.to_csv('out/metrics_model_selection/models_uni.csv')