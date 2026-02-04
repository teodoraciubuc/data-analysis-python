import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotly.figure_factory._dendrogram import sch
from sklearn.preprocessing import StandardScaler

df_Air=pd.read_csv("CalitateaAeruluiTari.csv")
df_Coduri=pd.read_csv("CoduriTari.csv")

#faci coloane/indicatori/ani
#conditie
#faci df_final1 si afisezi
indicatori = [
    "Air_quality_Carbon_Monoxide",
    "Air_quality_Ozone",
    "Air_quality_Nitrogen_dioxide",
    "Air_quality_Sulphur_dioxide",
    "Air_quality_PM2.5",
    "Air_quality_PM10"
]
#a1
#faci coloane/indicatori/ani
#conditie
#faci df_final1=pd.DataFrame({"Col": ceContine})(asta daca nu ai rasp pe randurile deja) si afisezi
maxim=df_Air[indicatori].idxmax()
tari=df_Air.loc[maxim,"Country"].values
df_final=pd.DataFrame({
    "Indicatori": indicatori,
    "Tari": tari,
})
df_final.to_csv("Cerinat1.csv",index=False)

print("Gata")

#a2
#merge+groupby in functie de care are mai multe din col
#avem conditie
#unesti merge [conditie, coloane]
#
df_rezultat=df_Air.merge(df_Coduri[["CountryID","Continent"]],left_on="CountryId",right_on="CountryID",how="left")
index_maxim = df_rezultat.groupby('Continent')[indicatori].idxmax()
cerinta2= index_maxim.reset_index()
for ind in indicatori:
    cerinta2[ind] = df_rezultat.loc[index_maxim[ind], "Country"].values
cerinta2.to_csv('Cerinta2.csv',index=False)
print("Gata")


#B

df_lucru = df_Air.set_index('Country')[indicatori]
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

df_popt.to_csv("popt.csv")

#C

R = pd.read_csv("g21_1.csv", index_col=0, decimal=",").to_numpy(dtype=float)
P = pd.read_csv("g21_2.csv", index_col=0, decimal=",").to_numpy(dtype=float)

R2 = R ** 2
P2 = P ** 2
mask = np.ones(R2.shape) - np.eye(R2.shape[0])
numitorul = np.sum(R2 * mask)
numitorul_plus_partial = numitorul + np.sum(P2 * mask)
kmo_global = numitorul / numitorul_plus_partial
print(f"Indexul KMO global este: {kmo_global}")

