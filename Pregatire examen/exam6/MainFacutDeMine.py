import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_Mortalitate = pd.read_csv("dataIN/Mortalitate.csv")
df_CoduriTari = pd.read_csv("dataIN/CoduriTariExtins.csv")

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(df_Mortalitate)
nan_replace_df(df_CoduriTari)

# ---------------- A1: tari cu RS negativ ----------------
cerinta1 = df_Mortalitate[df_Mortalitate["RS"] < 0][["Tara", "RS"]]
cerinta1.to_csv("./dataOUT/cerinta1.csv", index=False)

# ---------------- A2: valori medii pe continente ----------------
df_final = df_CoduriTari.merge(df_Mortalitate, left_on="Tari", right_on="Tara", how="inner")

indicatori = df_Mortalitate.columns[1:]          # RS, FR, IM, MMR, LE, LEM, LEF
cerinta2 = df_final.groupby("Continent")[indicatori].mean()
cerinta2.to_csv("./dataOUT/cerinta2.csv", index=True)

# ---------------- B: PCA standardizata ----------------
X = df_Mortalitate.iloc[:, 1:].values            # doar indicatorii (fara Tara)
X_std = StandardScaler().fit_transform(X)

pca = PCA()
c = pca.fit_transform(X_std)

n, m = X_std.shape
variatii = pca.explained_variance_
alpha = variatii * (n - 1) / n

# B1: variatiile componentelor (afisare la consola)
print("Variatii (explained_variance_):", variatii)
print("Alpha (corectat):", alpha)

# B2: scoruri (componente principale) -> scoruri.csv
scoruri = pd.DataFrame(
    c,
    index=df_Mortalitate["Tara"],
    columns=[f"CP{i}" for i in range(1, c.shape[1] + 1)]
)
scoruri.to_csv("./dataOUT/scoruri.csv", index=True)

# B3: grafic scoruri pe primele 2 axe principale
plt.figure(figsize=(8, 6))
plt.scatter(c[:, 0], c[:, 1])
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.title("Scoruri in primele 2 axe principale (CP1, CP2)")
plt.axhline(0)
plt.axvline(0)
plt.show()
