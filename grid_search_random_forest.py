#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
import sklearn
import xgboost
#%%
df = pd.read_csv('train_credit.csv')
test_df = pd.read_csv('test_credit.csv')
df.info()
# %%
x_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
       'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
y_column = 'default_0'
# %%
def mean_revenue_score(y_valid, y_predict):
    return (
        (y_predict == 0) * (
            1500 * (y_valid == 0) 
          - 5000 * (y_valid == 1))
    ).mean()
# %%
from sklearn.model_selection import train_test_split
from prepare_x import prepare_X

X = df[x_columns].copy()
X = prepare_X(X, fit=True)

y = df[y_column].copy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, random_state=0
)
# %%
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, make_scorer
beta = 5000 / 1500
scoring = {
    'fbeta': make_scorer(fbeta_score, beta=beta),
    'f1': make_scorer(f1_score),
    'mean_revenue': make_scorer(mean_revenue_score),
    'accuracy': make_scorer(accuracy_score)
}
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from datetime import datetime
start_t = datetime.now()
params = {
    'n_estimators': [ 50*i for i in range(2, 6) ],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [10*i for i in range(1, 4)],
    'min_samples_split': [ 10*x for x in range(7, 11)],
    'class_weight': [ 'balanced' ],
    'max_features': [6, 7, 8],
    'max_samples': [ 0.3, 0.4, 0.5 ],
    # 'max_depth': [2, 4, 8],
}
random_forest = RandomForestClassifier(random_state=1)
gscv = GridSearchCV(
    random_forest,
    params,
    scoring=scoring,
    refit='mean_revenue',
    n_jobs=-1,
    cv=5,
    verbose=1
)
gscv.fit(X, y)
print(gscv.best_params_)
print(gscv.best_score_)
print(datetime.now() - start_t)
# %%
best_params = {
    'class_weight': 'balanced', 
    'criterion': 'gini', 
    'max_features': 6, 
    'max_samples': 0.4, 
    'min_samples_leaf': 20, 
    'min_samples_split': 80, 
    'n_estimators': 100
}
scores = cross_validate(
    RandomForestClassifier(random_state=1, **best_params),
    X,
    y,
    scoring=scoring,
    cv=5,
    n_jobs=-1
)
print('       fbeta', f"{scores['test_fbeta'].mean():.5f}", scores['test_fbeta'].std(),)
print('          f1', f"{scores['test_f1'].mean():.5f}", scores['test_f1'].std())
print('mean_revenue', f"{scores['test_mean_revenue'].mean():.5f}", scores['test_mean_revenue'].std())
print('    accuracy', f"{scores['test_accuracy'].mean():.5f}", scores['test_accuracy'].std())
# %%
test_X = test_df[x_columns].copy()
test_X = prepare_X(test_X)

tree = RandomForestClassifier(random_state=1, **best_params)
tree.fit(X, y)
result = (1 - tree.predict(test_X))
print(*(1 - tree.predict(test_X)))