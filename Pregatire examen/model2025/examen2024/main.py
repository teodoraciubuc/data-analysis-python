import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from factor_analyzer import FactorAnalyzer

df_a = pd.read_csv("a.csv")
df_Coduri_Localitati = pd.read_csv("Coduri_Localitati.csv")
df_diversitatea = pd.read_csv("Diversitate.csv")

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(df_a)
nan_replace_df(df_Coduri_Localitati)
nan_replace_df(df_diversitatea)

coloane = list(df_diversitatea.columns[2:])
conditie = (df_diversitatea[coloane] == 0).any(axis=1)
df_Cerinta1 = df_diversitatea[conditie]
df_Cerinta1.to_csv("Cerinta1_incercare.csv", index=False)

df_final = df_diversitatea.merge(df_Coduri_Localitati[['Siruta', 'Judet']], on='Siruta')
df_final['Media'] = df_final[coloane].mean(axis=1)
idx_maxim = df_final.groupby('Judet')['Media'].idxmax()
df_rezultat = df_final.loc[idx_maxim, ['Judet', 'Localitate', 'Media']]
df_rezultat.to_csv("Cerinta2_incercare.csv", index=False)

def standardizare_df(t: pd.DataFrame, scal=True, ddof=0):
    x = t.values.astype(float)
    x_std = x - np.mean(x, axis=0)
    if scal:
        x_std = x_std / np.std(x, axis=0, ddof=ddof)
    return pd.DataFrame(x_std, index=t.index, columns=t.columns)

def tabelare_varianta_fact(varianta):
    t = pd.DataFrame(
        data={
            "Varianta": varianta[0],
            "Varianta cumulata": np.cumsum(varianta[0]),
            "Procent varianta": varianta[1] * 100,
            "Procent cumulat": varianta[2] * 100
        },
        index=["F" + str(i) for i in range(1, len(varianta[0]) + 1)]
    )
    t.index.name = "Factor"
    return t

X_std_df = standardizare_df(df_diversitatea[coloane], scal=True, ddof=0)

fa0 = FactorAnalyzer(rotation=None)
fa0.fit(X_std_df.values)
ev, _ = fa0.get_eigenvalues()
nr_factori = int((ev > 1).sum())
nr_factori = max(nr_factori, 1)

fa = FactorAnalyzer(n_factors=nr_factori, rotation="varimax")
fa.fit(X_std_df.values)

varianta = fa.get_factor_variance()
t_varianta = tabelare_varianta_fact(varianta)
t_varianta.to_csv("CerintaB1_Varianta.csv")
print(t_varianta.to_string())

r = pd.DataFrame(
    fa.loadings_,
    index=coloane,
    columns=[f"F{i}" for i in range(1, nr_factori + 1)]
)
r.to_csv("r.csv")
print(r.head().to_string())

print("B1 gata")
print("B2 gata")

#C

A = pd.read_csv("a.csv").to_numpy(dtype=float)
lam1 = 3.019
lam2 = 1.2203
h2 = lam1 * (A[:, 0]**2) + lam2 * (A[:, 1]**2)
for i in range(5):
    if h2[i] > 0.90:
        print(f"X{i+1}")
