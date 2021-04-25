#%%
import pandas as pd
import numpy as np
import seaborn as sns
#%%
df = pd.read_csv('train_credit.csv', index_col='ID')
test_df = pd.read_csv('test_credit.csv', index_col='ID')
df.info()
#%%
df.head()
# %%
sns.histplot(df.default_0);
# %%
df.columns
# %%
x_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
       'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
y_column = 'default_0'
# %%
from tqdm import trange
from typing import Type
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split

def mean_revenue_score(y_valid, y_predict):
    return ((y_predict == 0) * (1500 * (y_valid == 0) - 5000 * (y_valid == 1))).mean()

beta = 5000 / 1500
scoring = {
    'fbeta': make_scorer(fbeta_score, beta=beta),
    'f1': make_scorer(f1_score),
    'mean_revenue': make_scorer(mean_revenue_score),
    'accuracy': make_scorer(accuracy_score)
}
# %%
def total_delay(row):
    delay_columns = [f'PAY_{i}' for i in range(1, 7)]
    total = 0
    for column in delay_columns:
        if row[column] > 0:
            total += row[column]
    return total

def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    X['DELAY_TOTAL'] = X.apply(total_delay, axis=1)
    age_step = 10
    for age in range(20, 80, age_step):
        X[f'AGE_{age}_{age+age_step-1}'] = (X.AGE >= age) & (X.AGE < age + age_step)
    X.drop(columns=['AGE'], inplace=True)

    # pay delay in i-th month
    # is person pay in i-th month
    for i in range(1, 7):
        X[f'PAYED_{i}'] = X[f'PAY_{i}'] <= 0
        X[f'DELAY_{i}'] = X[f'PAY_{i}'] * (X[f'PAY_{i}'] > 0)
        X.drop(columns=[f'PAY_{i}'], inplace=True)

    X['IS_MALE'] = X.SEX == 1
    X['IS_FEMALE'] = X.SEX == 2
    X.drop(columns=['SEX'], inplace=True)

    X['HAS_SCHOOL'] = X['EDUCATION'] <= 3
    X['HAS_BACHELOR'] = X['EDUCATION'] <= 2
    X['HAS_MASTER'] = X['EDUCATION'] <= 1
    X.drop(columns=['EDUCATION'], inplace=True)
    
    one_hot_columns = ['MARRIAGE']
    for column in one_hot_columns:
        for value in X[column].unique():
            X[f'{column}_{value}'] = X[column] == value
        X.drop(columns=[column], inplace=True)

    # ratio of pay and limit
    # ratio of bill and limit
    # ratio of available credit and limit
    for i in range(1, 7):
        X[f'PAY_LR_{i}'] = X[f'PAY_AMT{i}'] / X['LIMIT_BAL']
        X[f'BILL_LR_{i}'] = X[f'BILL_AMT{i}'] / X['LIMIT_BAL']
        X[f'AVBLE_LR_{i}'] = 1 - X[f'BILL_AMT{i}'] / X['LIMIT_BAL']
        X.drop(columns=[f'PAY_AMT{i}'], inplace=True)
        X.drop(columns=[f'BILL_AMT{i}'], inplace=True)

    return X.drop(
        columns=[
            'AGE_20_29',
            'AGE_30_39',
            'AGE_40_49',
            'AGE_50_59',
            'AGE_60_69',
            'AGE_70_79',
            'PAYED_1',
            'PAYED_2',
            'DELAY_2',
            'PAYED_3',
            'DELAY_3',
            'PAYED_4',
            'DELAY_4',
            'PAYED_5',
            'DELAY_5',
            'PAYED_6',
            'DELAY_6',
            'IS_MALE',
            'IS_FEMALE',
            'HAS_SCHOOL',
            'HAS_BACHELOR',
            'HAS_MASTER',
            'MARRIAGE_1',
            'MARRIAGE_2',
            'MARRIAGE_3',
            'MARRIAGE_0'
        ])

X_base = prepare_X(df[x_columns].copy())
y_base = df[y_column].copy()

def my_train_test_split():
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_base, y_base, train_size=0.8
    )
    return X_train, X_valid, y_train, y_valid

X = X_base.copy()
y = y_base.copy()
# %%
def model_score(model: Type, params: dict, n_tries: int = 10):
    scores = {
        'fbeta': np.zeros(n_tries),
        'roc_auc': np.zeros(n_tries),
        'f1': np.zeros(n_tries),
        'mean_revenue': np.zeros(n_tries),
        'accuracy': np.zeros(n_tries)
    }
    for i in trange(n_tries):
        X_train, X_valid, y_train, y_valid = my_train_test_split()
        model_instance = model(**params, n_jobs=-1)
        model_instance.fit(X_train, y_train)
        y_predicted = model_instance.predict(X_valid)
        scores['fbeta'][i] = fbeta_score(y_valid, y_predicted, beta=beta)
        scores['roc_auc'][i] = roc_auc_score(y_valid, y_predicted)
        scores['f1'][i] = f1_score(y_valid, y_predicted)
        scores['mean_revenue'][i] = mean_revenue_score(y_valid, y_predicted)
        scores['accuracy'][i] = accuracy_score(y_valid, y_predicted)
    print()
    for score, results in scores.items():
        print(f'{score:>14}: {results.mean():8.3f} {results.std():8.3f}')
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

params = {

}

model_score(RandomForestClassifier, params)

# %%
import eli5
from eli5.sklearn import PermutationImportance

X_train, X_valid, y_train, y_valid = my_train_test_split()

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

perm = PermutationImportance(random_forest, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
# %%
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import numpy as np
plt.figure(figsize=(5,11))
importances = pd.Series(
    random_forest.feature_importances_, 
    X_train.columns
)
importances.sort_values()[:].plot.barh(
    color = cm.rainbow(np.linspace(-1.1,1.5))
);
# %%
column_to_drop = list(importances.index[importances < 0.03])
column_to_drop
# %%
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid
from datetime import datetime
from tqdm import tqdm
start_t = datetime.now()
params_grid = {
    'learning_rate': [0.01],
    # 'learning_rate': [0.005, 0.01, 0.02],

    'n_estimators': [ 250 ],
    # 'n_estimators': [ 50*i for i in range(2, 6) ],
    
    'reg_lambda': [ 0.2 ],
    # 'reg_lambda': [ 0, 0.2, 0.4, 0.6, 0.8, 1. ],

    'reg_alpha': [ 0 ],
    # 'reg_alpha': [ 0, 0.2, 0.4, 0.6, 0.8, 1. ],
    
    'colsample_bytree': [ 0.8 ],
    # 'colsample_bytree': [ 0.2, 0.4, 0.6, 0.8, 1. ],
    
    'subsample': [ 1. ],
    # 'subsample': [ 0.2, 0.4, 0.6, 0.8, 1. ],
    'subsample_freq': [ 1 ],
    
    'max_depth': [ 8 ],
    # 'max_depth': [i for i in range(3, 11)],
    
    'class_weight': [ 'balanced' ],
}
best_params = {}
best_score = -1e9
best_score_std = 0
for params in tqdm(list(ParameterGrid(params_grid))):
    scores = cross_val_score(
        LGBMClassifier(**params, n_jobs=-1), 
        X, 
        y, 
        scoring=make_scorer(mean_revenue_score),
        cv=10
    )
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_score_std = scores.std()
        best_params = params

print()
print(best_params)
print(best_score, best_score_std)
print(datetime.now() - start_t)
# %%
model_score(LGBMClassifier, best_params)
# %%
test_X = test_df[x_columns].copy()
test_X = prepare_X(test_X)

tree = LGBMClassifier(random_state=0, **best_params, n_jobs=-1)
tree.fit(X, y)
result = (1 - tree.predict(test_X))
print(*result)
# %%
proba = tree.predict_proba(test_X)
bad = np.argsort(proba[:,0], )[:3]
good = np.argsort(proba[:,1], )[:3]
print('bad')
print(proba[bad])
print(*test_df.index[bad])
print()
print('good')
print(proba[good])
print(*test_df.index[good])