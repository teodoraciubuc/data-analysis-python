import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly.figure_factory._dendrogram import sch
from sklearn.preprocessing import StandardScaler

df_Coduri=pd.read_csv("Coduri_Localitati.csv")
df_Diversitate=pd.read_csv("Diversitate.csv")
df_g20=pd.read_csv("g20.csv")

ani=list(df_Diversitate.columns[2:])
df_Diversitate["Media"]=df_Diversitate[ani].mean(axis=1)
df_final1=pd.DataFrame({"Siruta":df_Diversitate["Siruta"].values,
                        "Localitate":df_Diversitate["Localitate"].values,
                        "Media":df_Diversitate["Media"].values})
out=df_final1.sort_values("Media", ascending=False)
out.to_csv("Cerinta1.csv",index=False)


#a2
df_Cerinta1 = df_Diversitate[ani]

df_rezultat=df_Diversitate.merge(df_Coduri[["Siruta","Judet"]],left_on="Siruta",right_on="Siruta",how="left")
cerinta2 = df_rezultat.groupby('Judet')[ani].apply(lambda x: (x == 0).sum())
cerinta2.to_csv("Cerinta2.csv",index=True)
print("Gata")

#B
df_lucru = df_Diversitate.set_index('Siruta')[ani]
X = df_lucru.values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#Matricea
matrice = sch.linkage(X_std, method='ward')
print("Matricea:\n", matrice)

dist_agregare = matrice[:, 2]
difere_dist = np.diff(dist_agregare)
k_optim = len(dist_agregare) - np.argmax(difere_dist)
print("Numar optim clusteri (Elbow):", k_optim)

# Dendrograma
prag = (dist_agregare[len(dist_agregare) - k_optim] + dist_agregare[len(dist_agregare) - k_optim - 1]) / 2
plt.figure(figsize=(10, 6))
sch.dendrogram(matrice, labels=df_lucru.index, leaf_rotation=90)
plt.axhline(y=prag, color='r', linestyle='--', label=f'Prag optim (k={k_optim})')
plt.title("Dendrograma (Ward)")
plt.legend()
plt.tight_layout()
plt.show()

#Componenta partitie si salvare
k_ales = k_optim
labels_k = sch.fcluster(matrice, t=k_ales, criterion='maxclust')
df_popt = pd.DataFrame(index=df_lucru.index)
df_popt['Cluster'] = labels_k
df_popt.to_csv("CerintaB.csv")
#C
loadings = df_g20.iloc[:, 1:]
comunalitati = (loadings ** 2).sum(axis=1)
psi = 1 - comunalitati
print("Raspunsul e:", int(np.argmax(psi.values)+1))