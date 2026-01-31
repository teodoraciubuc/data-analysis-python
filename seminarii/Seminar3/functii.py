import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                t.fillna({c:t[c].mean()},inplace=True)
            else:
                t.fillna( {c:t[c].mode()[0]},inplace=True)

def calcul_procente(t:pd.Series):
    return t*100/t.sum()

def calcul_disim(t:pd.DataFrame):
    print(t)
    exit(0)
