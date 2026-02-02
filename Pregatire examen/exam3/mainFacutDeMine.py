import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import factor_analyzer as fa
from pandas.core.dtypes.common import is_numeric_dtype
from plotly.figure_factory._dendrogram import sch
from sklearn.preprocessing import StandardScaler


df_AirQuality = pd.read_csv("dataIN/AirQuality.csv", index_col=0)
df_Country = pd.read_csv("dataIN/CountryContinents.csv", index_col=0)

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(df_Country)
nan_replace_df(df_AirQuality)

# Cerinta A1
coloane = list(df_AirQuality.columns[1:])
maxim = df_AirQuality.set_index("Country")[coloane].idxmax()
df_rezultat = maxim.reset_index()
df_rezultat.columns = ["Indicator", "Country"]
df_rezultat.to_csv("dataOUT/Cerinta1.csv", index=False)
print("Gata!")

# Cerinta A2
df_final = df_Country.merge(df_AirQuality, left_index=True, right_index=True)
idx_max = df_final.groupby("Continent")[coloane].idxmax()
tari_max = idx_max.map(lambda v: df_Country.loc[v, "Country"])
cerinta2 = tari_max.reset_index()
cerinta2.to_csv("dataOUT/Cerinta2.csv", index=False)

#Cerinta B
#B1-Matricea ierarhie cu informatii privind jonctiunile si clusterii ,distanata dintre ei si nr de instante
X = df_AirQuality[coloane].values
X_std = StandardScaler().fit_transform(X)

matrice = sch.linkage(X_std, method="ward")
print("Matricea:\n", matrice)
dist = matrice[:, 2]
dif = np.diff(dist)
poz = int(np.argmax(dif))
prag = (dist[poz] + dist[poz + 1]) / 2
k_opt = X_std.shape[0] - (poz + 1)

#B2-Denograma pt partitia optimala
plt.figure(figsize=(10, 6))
sch.dendrogram(matrice, labels=df_AirQuality["Country"].values, color_threshold=prag)
plt.axhline(prag, color="r", linestyle="--")
plt.title(f"Dendrograma Ward (k_opt={k_opt})")
plt.tight_layout()
plt.show()

#B3-COMPONENTA PARTITIEI Optimale
k_ales = k_opt
labels_k = sch.fcluster(matrice, t=k_ales, criterion="maxclust")

df_popt = pd.DataFrame({"Cluster": labels_k})
df_popt["Country"] = df_AirQuality["Country"].values
df_popt = df_popt.set_index("Country")

df_popt.to_csv("dataOUT/popt.csv")
print("popt.csv gata!")

#C
R = pd.read_csv("dataIN/FISIER_CORELATII.csv", index_col=0).to_numpy(dtype=float)
P = pd.read_csv("dataIN/FISIER_CORELATII_PARTIALE.csv", index_col=0).to_numpy(dtype=float)
n = R.shape[0]
iu = np.triu_indices(n, k=1)
kmo_global = np.sum(R[iu] ** 2) / (np.sum(R[iu] ** 2) + np.sum(P[iu] ** 2))
print("KMO global =", kmo_global)