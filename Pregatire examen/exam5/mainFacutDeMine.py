import scipy.cluster.hierarchy as hic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from factor_analyzer import FactorAnalyzer
from pandas.core.dtypes.common import is_numeric_dtype
from plotly.figure_factory._dendrogram import sch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

df_alcohol=pd.read_csv('dataIN/alcohol.csv')
df_CoduriTari=pd.read_csv('dataIN/CoduriTariExtins.csv')

nan_replace_df(df_alcohol)
nan_replace_df(df_CoduriTari)

#A1
ani = ['2000', '2005', '2010', '2015', '2018']
df_alcohol['Consum_Mediu'] = df_alcohol[ani].mean(axis=1)
df_rez = df_alcohol.sort_values(by='Consum_Mediu', ascending=False)
cerinta1 = df_rez[['Code', 'Country', 'Consum_Mediu']]
cerinta1.to_csv('./dataOUT/Cerinta1.csv', index=False)

#A2
df_final=df_CoduriTari.merge(df_alcohol, left_on='Country', right_on='Country', how='left')
media=df_final.groupby('Continent')[ani].mean()
maxim=media.idxmax(axis=1)
df_rezultat=pd.DataFrame({'Continent':maxim.index,'Anul':maxim})
df_rezultat.to_csv('./dataOUT/Media.csv', index=False)

#B1
X = df_alcohol[ani].values
X_std = StandardScaler().fit_transform(X)

matrice = hic.linkage(X_std, method="ward")
print("Matricea:\n", matrice)

#B2
k = 5
etichete = hic.fcluster(matrice, t=k, criterion='maxclust')

p4 = pd.DataFrame({
    "Code": df_alcohol["Code"],
    "Country": df_alcohol["Country"],
    "Cluster": etichete
})

p4.to_csv("./dataOUT/p4.csv", index=False)
print("Gata p4.csv")

#B3
pca = PCA(n_components=2)
C = pca.fit_transform(X_std)

plt.figure()
plt.scatter(C[:, 0], C[:, 1], c=etichete)
plt.xlabel("C1")
plt.ylabel("C2")
plt.title("Partitia in 5 clusteri (axe = primele 2 CP)")
plt.show()

#C
x = np.array([3, 1, 2, 1, 4])

# Matricea A (vectori proprii)
A = df_alcohol.values
scores = np.dot(x, A)
print(f"Instanța x: {x}")
print("Scorurile obținute:",scores)
