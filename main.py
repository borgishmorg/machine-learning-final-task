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
df.info()
#%%
df.head()
# %%
sns.histplot(df.default_0);
# %%
df.columns
# %%
x_columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
       'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
y_column = 'default_0'
# %%
from sklearn.model_selection import train_test_split
X = df[x_columns].copy()

# X['BILL_ON_LIM1'] = X['BILL_AMT1'] / X['LIMIT_BAL']
# X['BILL_ON_LIM2'] = X['BILL_AMT2'] / X['LIMIT_BAL']
# X['BILL_ON_LIM3'] = X['BILL_AMT3'] / X['LIMIT_BAL']
# X['BILL_ON_LIM4'] = X['BILL_AMT4'] / X['LIMIT_BAL']
# X['BILL_ON_LIM5'] = X['BILL_AMT5'] / X['LIMIT_BAL']
# X['BILL_ON_LIM6'] = X['BILL_AMT6'] / X['LIMIT_BAL']

# X['PAY_ON_LIM1'] = X['PAY_AMT1'] / X['LIMIT_BAL']
# X['PAY_ON_LIM2'] = X['PAY_AMT2'] / X['LIMIT_BAL']
# X['PAY_ON_LIM3'] = X['PAY_AMT3'] / X['LIMIT_BAL']
# X['PAY_ON_LIM4'] = X['PAY_AMT4'] / X['LIMIT_BAL']
# X['PAY_ON_LIM5'] = X['PAY_AMT5'] / X['LIMIT_BAL']
# X['PAY_ON_LIM6'] = X['PAY_AMT6'] / X['LIMIT_BAL']

y = df[y_column].copy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, random_state=0
)
print(y_train.mean(), y_valid.mean())
# %%
def mean_revenue_score(y_valid, y_predict):
    return ((y_predict == 0) * (1500 * (y_valid == 0) - 5000 * (y_valid == 1))).mean()
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, fbeta_score

beta = 5000 / 1500

random_forest = RandomForestClassifier(random_state=1)
random_forest.fit(X_train, y_train)
print('roc_auc', roc_auc_score(y_valid, random_forest.predict(X_valid)))
print('f1', f1_score(y_valid, random_forest.predict(X_valid)))
print('fbeta', fbeta_score(y_valid, random_forest.predict(X_valid), beta=beta))
print('accuracy', accuracy_score(y_valid, random_forest.predict(X_valid)))
print('mean_revenue', mean_revenue_score(y_valid, random_forest.predict(X_valid)))
# %%
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(random_forest, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
# %%
dummy_random_forest = RandomForestClassifier(random_state=1)
dummy_random_forest.fit(X_train[['PAY_1']], y_train)
print('roc_auc', roc_auc_score(y_valid, dummy_random_forest.predict(X_valid[['PAY_1']])))
print('f1', f1_score(y_valid, dummy_random_forest.predict(X_valid[['PAY_1']])))
print('fbeta', fbeta_score(y_valid, dummy_random_forest.predict(X_valid[['PAY_1']]), beta=beta))
print('accuracy', accuracy_score(y_valid, dummy_random_forest.predict(X_valid[['PAY_1']])))
print('mean_revenue', mean_revenue_score(y_valid, dummy_random_forest.predict(X_valid[['PAY_1']])))
# %%
