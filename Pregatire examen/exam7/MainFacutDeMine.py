import os
import scipy.cluster.hierarchy as hic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


df_alcohol = pd.read_csv("dataIN/alcohol.csv")
df_CoduriTari = pd.read_csv("dataIN/CoduriTariExtins.csv")

nan_replace_df(df_alcohol)
nan_replace_df(df_CoduriTari)

ani = ["2000", "2005", "2010", "2015", "2018"]

# ---------------- A1 ----------------
df_alcohol["Media"] = df_alcohol[ani].mean(axis=1)
cerinta1 = df_alcohol[["Code", "Media"]]
cerinta1.to_csv("dataOUT/cerinta1.csv", index=False)

# ---------------- A2 ----------------
df_final = df_CoduriTari.merge(df_alcohol, left_on="Tari", right_on="Entity", how="inner")
media_continent = df_final.groupby("Continent")[ani].mean()
an_max = media_continent.idxmax(axis=1)

cerinta2 = pd.DataFrame({
    "Continent_Name": an_max.index,
    "Anul": an_max.values
})
cerinta2.to_csv("dataOUT/cerinta2.csv", index=False)

# ---------------- B1 ----------------
X = df_alcohol[ani].values
X_std = StandardScaler().fit_transform(X)

matrice = hic.linkage(X_std, method="ward")
print("Matricea ierarhie:\n", matrice)
dist = matrice[:, 2]
dif = np.diff(dist)
poz = int(np.argmax(dif))
prag = (dist[poz] + dist[poz + 1]) / 2

# ---------------- B2 ----------------
plt.figure(figsize=(12, 6))
hic.dendrogram(matrice, labels=df_alcohol["Code"].values)
plt.axhline(prag, linestyle="--")
plt.title("Dendrograma (Ward) - partitie optima")
plt.show()

# ---------------- B3 ----------------
etichete_opt = hic.fcluster(matrice, t=prag, criterion="distance")

popt = pd.DataFrame({
    "Code": df_alcohol["Code"],
    "Country": df_alcohol["Entity"],
    "Cluster": etichete_opt
})
popt.to_csv("dataOUT/popt.csv", index=False)
print("Gata popt.csv")
