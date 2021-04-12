import pandas as pd


def total_delay(row):
    delay_columns = [f'PAY_{i}' for i in range(1, 7)]
    total = 0
    for column in delay_columns:
        if row[column] > 0:
            total += row[column]
    return total


def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    X['DELAY_COUNT'] = X.apply(total_delay, axis=1)
    X.loc[X['EDUCATION'] > 3, 'EDUCATION'] = 4
    X.loc[X['PAY_1'] < 0, 'PAY_1'] = 0
    X.loc[X['PAY_2'] < 0, 'PAY_2'] = 0
    X.loc[X['PAY_3'] < 0, 'PAY_3'] = 0
    X.loc[X['PAY_4'] < 0, 'PAY_4'] = 0
    X.loc[X['PAY_5'] < 0, 'PAY_5'] = 0
    X.loc[X['PAY_6'] < 0, 'PAY_6'] = 0

    step = 5
    for age in range(20, 80, step):
        X[f'AGE_{age}_{age+step}'] = (X['AGE'] >= age) & (X['AGE'] < age + step)

    X.drop('AGE', axis=1, inplace=True)

    X['BILL_ON_LIM1'] = X['BILL_AMT1'] / X['LIMIT_BAL']
    X['BILL_ON_LIM2'] = X['BILL_AMT2'] / X['LIMIT_BAL']
    X['BILL_ON_LIM3'] = X['BILL_AMT3'] / X['LIMIT_BAL']
    X['BILL_ON_LIM4'] = X['BILL_AMT4'] / X['LIMIT_BAL']
    X['BILL_ON_LIM5'] = X['BILL_AMT5'] / X['LIMIT_BAL']
    X['BILL_ON_LIM6'] = X['BILL_AMT6'] / X['LIMIT_BAL']

    X['PAY_ON_LIM1'] = X['PAY_AMT1'] / X['LIMIT_BAL']
    X['PAY_ON_LIM2'] = X['PAY_AMT2'] / X['LIMIT_BAL']
    X['PAY_ON_LIM3'] = X['PAY_AMT3'] / X['LIMIT_BAL']
    X['PAY_ON_LIM4'] = X['PAY_AMT4'] / X['LIMIT_BAL']
    X['PAY_ON_LIM5'] = X['PAY_AMT5'] / X['LIMIT_BAL']
    X['PAY_ON_LIM6'] = X['PAY_AMT6'] / X['LIMIT_BAL']

    return X