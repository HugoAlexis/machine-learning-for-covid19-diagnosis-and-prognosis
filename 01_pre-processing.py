import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imputation import GroupsImputer


NANS_TO_DELATE_COL = 30
NTEST_PER_GROUP = 12
RANDOM_STATE = 8


# Carga el dataset original
data = (
    pd.read_excel('../data/Immunometabolism_COVID-19_Data.xlsx')
    .drop(columns=['ID '])
)
data['Positivo'] = (data['Group']>1).astype(int)
data.columns = data.columns.str.replace(' ', '_')

# Trnsforma las columnas al tipo de dato correcto
columns_object = data.columns[data.dtypes=='object']
columns_object_del = []

for col in columns_object:
    data[col] = data[col].str.replace('<', '')
    data[col] = data[col].str.replace('>', '')
    
    try:
        data[col] = data[col].astype(float)
    except ValueError:
        columns_object_del.append(col)

# Elimina las columnas con muchos datos faltantes
nulls_per_column = data.isnull().sum(axis=0)
columns_del = list(nulls_per_column[nulls_per_column>=NANS_TO_DELATE_COL].index) \
              + columns_object_del

data_del = data[columns_del]
data.drop(columns=columns_del, inplace=True);


## Separa en Train-Test

groups = data.Group

train_index = []
test_index = []

rdm = np.random.RandomState(RANDOM_STATE)

for group in [1, 2, 3, 4]:
    index_group = data[data.Group==group].index
    test_index_group = rdm.choice(a      = index_group, 
                                  size   = NTEST_PER_GROUP, 
                                  replace=False)    
    test_index.extend(test_index_group)


train_index = data.index.difference(test_index)
train_index = np.array(train_index)
test_index = np.array(test_index)

data_train = data.loc[train_index]
data_test  = data.loc[test_index]


# Crea la imputación de datos
filler = GroupsImputer()
filler.fit(data_train, data_train.Group)

data_train_filled = filler.transform(data_train, data_train.Group)
data_test_filled = filler.transform(data_test, data_test.Group)

# Guardar datasets
data_train_filled.to_csv('data_preprocess/tran.csv', index=False)
data_test_filled.to_csv('data_preprocess/test.csv', index=False)





