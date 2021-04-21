#%%
import pandas as pd
import numpy as np
#%%
RANDOM_STATE = 6
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
from prepare_x import prepare_X

X = df[x_columns].copy()
X = prepare_X(X, fit=True)
y = df[y_column].copy()
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
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from datetime import datetime
start_t = datetime.now()
params = {
    'subsample_freq': [1],
    'subsample': [0.4],
    'colsample_bytree': [0.4],
    'reg_alpha': [0.4],
    'reg_lambda': [0.6],
    'max_depth': [5],
    'class_weight': [ 'balanced' ],
}
random_forest = LGBMClassifier(random_state=RANDOM_STATE)
gscv = GridSearchCV(
    random_forest,
    params,
    scoring=scoring,
    refit='mean_revenue',
    n_jobs=-1,
    cv=StratifiedKFold(),
    verbose=1
)
gscv.fit(X, y)
print(gscv.best_params_)
print(gscv.best_score_)
print(datetime.now() - start_t)
# %%
scores = cross_validate(
    LGBMClassifier(random_state=RANDOM_STATE, **gscv.best_params_),
    X,
    y,
    scoring=scoring,
    cv=StratifiedKFold(),
    n_jobs=-1
)
print('       fbeta', f"{scores['test_fbeta'].mean():.5f}", scores['test_fbeta'].std(),)
print('            ', f"{scores['test_fbeta'].min():.5f}", scores['test_fbeta'].max(),)
print('          f1', f"{scores['test_f1'].mean():.5f}", scores['test_f1'].std())
print('            ', f"{scores['test_f1'].min():.5f}", scores['test_f1'].max(),)
print('mean_revenue', f"{scores['test_mean_revenue'].mean():.5f}", scores['test_mean_revenue'].std())
print('            ', f"{scores['test_mean_revenue'].min():.5f}", scores['test_mean_revenue'].max(),)
print('    accuracy', f"{scores['test_accuracy'].mean():.5f}", scores['test_accuracy'].std())
print('            ', f"{scores['test_accuracy'].min():.5f}", scores['test_accuracy'].max(),)
# %%
test_X = test_df[x_columns].copy()
test_X = prepare_X(test_X)

tree = LGBMClassifier(random_state=RANDOM_STATE, **gscv.best_params_)
tree.fit(X, y)
print(*(1 - tree.predict(test_X)))

# %%
proba = tree.predict_proba(test_X)
bad = np.argsort(proba[:,0], )[:3]
good = np.argsort(proba[:,1], )[:3]
print('bad')
print(proba[bad])
print(test_df.iloc[bad].ID)
print()
print('good')
print(proba[good])
print(test_df.iloc[good].ID)