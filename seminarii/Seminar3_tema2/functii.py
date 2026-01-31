import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def calcul_procente(t: pd.Series):
    return t * 100 / t.sum()


def calcul_disim(t: pd.DataFrame):
    x = t.values
    tx = np.sum(x, axis=0)
    tx[tx == 0] = 1
    sx = np.sum(x, axis=1)
    r = (sx - x.T).T
    tr = np.sum(r, axis=0)
    d = 0.5 * np.sum(np.abs(x / tx - r / tr), axis=0)
    return pd.Series(d, t.columns)


def shannon(t: pd.DataFrame):
    x = t.values
    tx = np.sum(x, axis=0)
    tx[tx == 0] = 1
    p = x / tx
    p[p == 0] = 1
    h = -np.sum(p * np.log2(p), axis=0)
    return pd.Series(h, t.columns)
