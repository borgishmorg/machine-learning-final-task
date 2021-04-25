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
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, make_scorer
beta = 5000 / 1500
scoring = {
    'fbeta': make_scorer(fbeta_score, beta=beta),
    'f1': make_scorer(f1_score),
    'mean_revenue': make_scorer(mean_revenue_score),
    'accuracy': make_scorer(accuracy_score)
}
# %%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
params = {
    # 'loss': ['deviance', 'exponential'],
    # 'learning_rate': [0.2, 0.1, 0.01],
    'booster': [
        # 'gbtree',
        # 'dart', 
        'gblinear'
    ],
    # 'lambda': [0, 0.01, 0.1, 1],
    # 'alpha': [0, 0.01, 0.1, 1],

    # 'eta': [0.3],
    'eta': [0.01],
    # 'subsample': [1.],
    # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.],
    # 'max_delta_step': [0.8],
    # 'max_delta_step': [0.5, 0.6, 0.7, 0.8, 0.9, 1.],
    # 'max_depth': [2],
    # 'max_depth': range(1, 7),
    
    # 'min_child_weight': [5, 10, 15],
    # 'max_delta_step': [0.1],
    'eval_metric': ['auc'],
    'scale_pos_weight': [(y == 0).sum() / (y == 1).sum()],
    # 'n_estimators': [75, 100, 125],
    # 'criterion': ['friedman_mse'],
    # 'criterion': ['friedman_mse', 'mse'],
    
    # 'max_depth': [4, 8, 12, None],
    # 'max_features': ['sqrt', 'log2'],
    # 'max_features': [6, 7, 8]
}
random_forest = XGBClassifier(random_state=1, use_label_encoder=False)
gscv = GridSearchCV(
    random_forest,
    params,
    scoring=scoring,
    refit='mean_revenue',
    # n_jobs=-1,
    cv=5,
    verbose=1
)
gscv.fit(X, y)
print(gscv.best_params_)
print(gscv.best_score_)
# %%
scores = cross_validate(
    XGBClassifier(random_state=1, **gscv.best_params_, use_label_encoder=False),
    # ExtraTreesClassifier(random_state=1),
    X,
    y,
    scoring=scoring,
    cv=5
)
print('       fbeta', f"{scores['test_fbeta'].mean():.5f}", scores['test_fbeta'].std(),)
print('          f1', f"{scores['test_f1'].mean():.5f}", scores['test_f1'].std())
print('mean_revenue', f"{scores['test_mean_revenue'].mean():.5f}", scores['test_mean_revenue'].std())
print('    accuracy', f"{scores['test_accuracy'].mean():.5f}", scores['test_accuracy'].std())
# %%
import eli5
from eli5.sklearn import PermutationImportance

random_forest = XGBClassifier(random_state=1, use_label_encoder=False).fit(X_train, y_train)
perm = PermutationImportance(random_forest, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
# %%
test_X = test_df[x_columns].copy()
test_X = prepare_X(test_X)

tree = XGBClassifier(random_state=1, **gscv.best_params_, use_label_encoder=False)
tree.fit(X, y)
print(*(1 - tree.predict(test_X)))