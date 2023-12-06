import numpy as np


def mult_confusion_matrix(y_group, y_pred, r=0.5):
    y_group = np.array(y_group)
    y_pred = np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    y_true = np.array(y_group > 1)
    
    mat = np.zeros(shape=(2, 4))
    
    mat[0, 0] = ((y_group == 4) & y_pred == 1).sum()
    mat[1, 0] = np.sum(y_group == 4) - mat[0, 0]
    mat[0, 1] = ((y_group == 3) & y_pred == 1).sum()
    mat[1, 1] = np.sum(y_group == 3) - mat[0, 1]
    mat[0, 2] = ((y_group == 2) & y_pred == 1).sum()
    mat[1, 2] = np.sum(y_group == 2) - mat[0, 2]
    mat[0, 3] = ((y_group == 1) & y_pred == 1).sum()
    mat[1, 3] = np.sum(y_group == 1) - mat[0, 3]
    
    return mat


def ensemble_values(y_prob1, y_prob2, w=(0.5, 0.5)):
    w = np.array(w) / np.sum(w)
    y_prob1 = np.array(y_prob1)
    y_prob2 = np.array(y_prob2)
    
    y_prob = w[0]*y_prob1 + w[1]*y_prob2
    return y_prob

def ensemble_values3(y_prob1, y_prob2, y_prob3, w=(1, 1, 1)):
    w = np.array(w) / np.sum(w)
    y_prob1 = np.array(y_prob1)
    y_prob2 = np.array(y_prob2)
    y_prob3 = np.array(y_prob3)
    
    y_prob = w[0]*y_prob1 + w[1]*y_prob2 + w[2]*y_prob3
    return y_prob

def mult_confusion_matrix(y_group, y_pred, r=0.5):
    y_group = np.array(y_group)
    y_pred = np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    y_true = np.array(y_group > 1)
    
    mat = np.zeros(shape=(2, 4))
    
    mat[0, 0] = ((y_group == 4) & y_pred == 1).sum()
    mat[1, 0] = np.sum(y_group == 4) - mat[0, 0]
    mat[0, 1] = ((y_group == 3) & y_pred == 1).sum()
    mat[1, 1] = np.sum(y_group == 3) - mat[0, 1]
    mat[0, 2] = ((y_group == 2) & y_pred == 1).sum()
    mat[1, 2] = np.sum(y_group == 2) - mat[0, 2]
    mat[0, 3] = ((y_group == 1) & y_pred == 1).sum()
    mat[1, 3] = np.sum(y_group == 1) - mat[0, 3]
    
    return mat


def accuracy_score(y_true, y_pred, r=0.5):
    """
    Calcula la exactitud (accuracy) de los valores verdaderos y predichos por el 
    modelo para la clasificacion binaria.
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Accuracy (exactitud) de los valores reales y los valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    score = np.sum(y_true == y_pred) / len(y_true)
    return score

def true_positive(y_true, y_pred, r=0.5):
    """
    Devuelve el numero de verdaderos positivos (Numero de datos 
    cuyo valor es positivo (1), y el modelo acierta prediciendolos
    como positivo(1)).
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    TN: int
        Cantidad de verdaderos positivos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    score = np.array(y_true, dtype=bool) & np.array(y_pred, dtype=bool)
    return score.sum()
    
def true_negative(y_true, y_pred, r=0.5):
    """
    Devuelve el numero de verdaderos negativos (Numero de datos 
    cuyo valor es negativo (0), y el modelo acierta prediciendolos
    como negativo(0)).
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    TN: int
        Cantidad de verdaderos negativos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    y_true = np.array(y_true, dtype=bool), 
    y_pred = np.array(y_pred, dtype=bool)
    
    return true_positive(np.invert(y_true), 
                         np.invert(y_pred))

def false_negative(y_true, y_pred, r=0.5):
    """
    Devuelve el numero de falsos negativos (Numero de datos 
    cuyo valor es positivo (1), y el modelo falla prediciendolos
    como negativo(0)).
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    FN: int
        Cantidad de falsos negativos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    
    score = np.invert(y_pred) & y_true
    return score.sum()

def false_positive(y_true, y_pred, r=0.5):
    """
    Devuelve el numero de falsos positivos (Numero de datos 
    cuyo valor es negativo (0), y el modelo falla prediciendolos
    como positivo (1)).
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    FP: int
        Cantidad de falsos positivos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    
    score = y_pred & np.invert(y_true)
    return score.sum()

def confusion_matrix(y_true, y_pred, r=0.5):
    """
    Devuelve la matriz de confusion (para clasificacion binaria). 
    Con los valores predecidos en las filas, y los valores reales
    en las columnas.
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    mat: np.array(shape=(2, 2))
        Matriz de confusion para la clasificacion binaria.
    """
    if r == 'best':
        r = select_threshold(y_true, y_pred)
        print(f'r = {r}')
    
    mat = np.zeros(shape=(2, 2))
    mat[0, 0] = true_positive(y_true, y_pred, r=r)
    mat[0, 1] = false_positive(y_true, y_pred, r=r)
    mat[1, 0] = false_negative(y_true, y_pred, r=r)
    mat[1, 1] = true_negative(y_true, y_pred, r=r)
    
    return mat

def sensitivity(y_true, y_pred, r=0.5):
    """
    Calcula la sensitividad (probabilidad de que el valor predecido 
    sea positivo, dado que el valor real es positivo) de los valores 
    verdaderos y los valores predecidos por el modelo. 
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Sensitividad de los valores reales y los valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    score = true_positive(y_true, y_pred) / np.sum(y_true)
    return score

recall = sensitivity

def specificity(y_true, y_pred, r=0.5):
    """
    Calcula la especificidad (probabilidad de que el valor predecido 
    sea negativo, dado que el valor real es negativo) de los valores 
    verdaderos y los valores predecidos por el modelo. 
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Especificidad de los valores reales y los valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    score = true_negative(y_true, y_pred) / (len(y_true) - np.sum(y_true))
    return score

def precision(y_true, y_pred, r=0.5):
    """
    Calcula la precision (razon de verdaderos positivos, con respecto al total 
    de los valores positivos predecidos) de los valores verdaderos y los valores
    predecidos por el modelo. 
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Precision de los valores reales y los valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    TP, FP = true_positive(y_true, y_pred), false_positive(y_true, y_pred)
    score = (TP) / (TP + FP)
    return score

def f1_score(y_true, y_pred, r=0.5):
    """
    Media armonica de la precision y la sensitividad del modelo.
    
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Valor f (f1 score) de los valores reales y los valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    PPV = precision(y_true, y_pred)
    TPR = sensitivity(y_true, y_pred)
    score = 2 * (PPV * TPR) / (PPV + TPR)
    return score

def balanced_accuracy(y_true, y_pred, r=0.5):
    """
    Calcula la exactitud (accuracy) de los valores verdaderos y predichos por el 
    modelo para la clasificacion binaria, utilizando una media para balancear los 
    datos (util cuando el numero de instancias de cada clase es muy diferente).
    
    Parametros:
    ============
    y_true: list (array-like)
        Lista con los valores verdaderos
    y_pred: list (array-like)
        Lista con los valores predecidos.
    r: float (0 <= r <=1) (opcional)
        Si los valores predecidos son valores reales continuos 
        (probabilidades), los valores mayores a r se toman como 
        positivos (1) y los restantes como negativos (0).
        
    Return: 
    ============
    score: float
        Balanced Accuracy (exactitud balanceada) de los valores reales y los 
        valores predecidos.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.array(y_pred > r, dtype=int)
    
    TPR = sensitivity(y_true, y_pred)
    TNR = specificity(y_true, y_pred)
    score = (TPR + TNR) / 2
    return score

def get_metrics(TP, FP, FN, TN):
    precision =  TP/(TP+FP)
    sensitivity = TP/(TP + FN)
    recall =  TP/(TP+FN) 
    f1 = 2*precision*recall/(precision + recall)
    specificity = TN / (TN+FP)
    accuracy = (sensitivity + specificity)/2
    
    metrics = {
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'accuracy': accuracy, 
        'specificity': specificity}
    
    return metrics


def get_roc_curve(ytrue, ypred):
    sensitivies = [1]
    specificities = [0]
    
    dt=0.005
    for umbral in np.arange(dt, 1, dt):
        sensitivity_i = sensitivity(ytrue, ypred, r=umbral)
        specificity_i = specificity(ytrue, ypred, r=umbral)
        sensitivies.append(sensitivity_i)
        specificities.append(specificity_i)
        
    return specificities, sensitivies

def select_threshold(ytrue, ypred, metric=balanced_accuracy):
    r_best = None
    best_score = 0
    for ri in np.linspace(0, 1, 1000):
        y_class = np.array(ypred>ri, dtype=int)
        score = metric(ytrue, y_class)
        
        if (r_best is None) or score>best_score:
            r_best = ri
            best_score = score
        
    return r_best
