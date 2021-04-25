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
X = prepare_X(X)

y = df[y_column].copy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, random_state=0
)
# %%
beta = 5000 / 1500
scoring = {
    'fbeta': make_scorer(fbeta_score, beta=beta),
    'f1': make_scorer(f1_score),
    'mean_revenue': make_scorer(mean_revenue_score),
    'accuracy': make_scorer(accuracy_score)
}
# %%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, make_scorer
params = {
    'n_estimators': [75, 100, 125],
    'criterion': ['entropy'],
    # 'criterion': ['gini', 'entropy'],
    'min_samples_split': [ 45 + 5*x for x in range(1, 6)],
    # 'max_depth': [4, 8, 16],
    'class_weight': ['balanced'],
    # 'max_features': ['sqrt', 'log2']
    'max_features': [6, 7, 8]
}
random_forest = ExtraTreesClassifier(random_state=1)
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
# %%
scores = cross_validate(
    ExtraTreesClassifier(random_state=1, **gscv.best_params_),
    # ExtraTreesClassifier(random_state=1),
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
import eli5
from eli5.sklearn import PermutationImportance

random_forest = ExtraTreesClassifier(random_state=1).fit(X_train, y_train)
perm = PermutationImportance(random_forest, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
# %%
test_X = test_df[x_columns].copy()
test_X = prepare_X(test_X)

tree = ExtraTreesClassifier(random_state=1, **gscv.best_params_)
tree.fit(X, y)
result = 1 - tree.predict(test_X)
print(*(1 - tree.predict(test_X)))