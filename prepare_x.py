import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer

def total_delay(row):
    delay_columns = [f'PAY_{i}' for i in range(1, 7)]
    total = 0
    for column in delay_columns:
        if row[column] > 0:
            total += row[column]
    return total

transformer = None

def prepare_X(X: pd.DataFrame, fit=False) -> pd.DataFrame:
    X['DELAY_COUNT'] = X.apply(total_delay, axis=1)
    # X.loc[X['EDUCATION'] > 3, 'EDUCATION'] = 4
    # X.loc[X['PAY_1'] < 0, 'PAY_1'] = 0
    # X.loc[X['PAY_2'] < 0, 'PAY_2'] = 0
    # X.loc[X['PAY_3'] < 0, 'PAY_3'] = 0
    # X.loc[X['PAY_4'] < 0, 'PAY_4'] = 0
    # X.loc[X['PAY_5'] < 0, 'PAY_5'] = 0
    # X.loc[X['PAY_6'] < 0, 'PAY_6'] = 0

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

    global transformer
    if fit:
        transformer = ColumnTransformer(
            (
                # ('one_hot', OneHotEncoder(), [
                ('passthrough2', 'passthrough', [
                    'SEX', 'EDUCATION', 'MARRIAGE',
                    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
                ]),
                # ('min_max', StandardScaler(), [
                ('passthrough1', 'passthrough', [
                    'LIMIT_BAL',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                ]),
                ('age', KBinsDiscretizer(10), ['AGE']),
                ('passthrough', 'passthrough', ['DELAY_COUNT'])
            )
        )
        transformer.fit(X)
    
    return transformer.transform(X)