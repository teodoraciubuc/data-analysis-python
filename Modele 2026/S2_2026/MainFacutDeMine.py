import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
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

# 1. Salvarea în fișierul Cerinta1.csv a coeficienților de variație pentru fiecare indicator. Se va salva
# pentru fiecare indicator numele indicatorului și coeficientul de variație.


cv=df_Air[indicatori].std(ddof=1)/df_Air[indicatori].mean()
df_final1=cv.reset_index()
df_final1.columns = [["Indicator","CV"]]
df_final1.to_csv("Cerinta1.csv",index=False)
print("GATA")

# 2. Să se determine pentru fiecare continent indicatorul cu cel mai mare coeficient de variație.
# Datele vor fi salvate fișierul Cerinta2.csv. Pentru fiecare continent se va salva numele
# continentului, indicatorul cu coeficientul de variație cel mai mare și valoarea coeficientului de
# variație. (2 puncte)

df_rezultat=df_Air.merge(df_Coduri[["CountryID","Continent"]], left_on="CountryId", right_on="CountryID", how="left")

mean_conditie=df_rezultat.groupby("Continent")[indicatori].mean()
std_conditie=df_rezultat.groupby("Continent")[indicatori].std(ddof=1)
cv_contitie=std_conditie/mean_conditie
df_final=pd.DataFrame({"Continent":cv_contitie.index,
                       "Indicator":cv_contitie.idxmax(axis=1).values,
                       "CV":cv_contitie.max(axis=1).values
                       })
df_final.to_csv("Cerinta2.csv",index=False)

#B
df_pca=df_Air.set_index('CountryId')[indicatori].copy()

valori_numerice = df_pca[indicatori].values
indicatori_standardizati = StandardScaler().fit_transform(valori_numerice)
pca = PCA()
c = pca.fit_transform(indicatori_standardizati)
n, m = indicatori_standardizati.shape
variatii = pca.explained_variance_
alpha = variatii * (n-1)/n

#Variante
print('explained_variance (Sample): ', variatii)
print('alpha (Population): ', alpha)


# Scoruri
scoruri = pd.DataFrame(c, index=df_pca.index, columns=[f"CP{i}" for i in range(1, c.shape[1]+1)])
scoruri.to_csv('CerintaB_Scoruri.csv', index=True)



#Scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(c[:, 0], c[:, 1], c='b', alpha=0.6)
tari = df_pca.index
for i in range(len(tari)):
    plt.text(c[i, 0], c[i, 1], tari[i], fontsize=8)
plt.xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("Scoruri in primele 2 axe principale (CP1, CP2)")
plt.axhline(0, c='k', linestyle='--')
plt.axvline(0, c='k', linestyle='--')
plt.grid()
plt.show()


#C

valori_proprii = [0.9, 0.05, 0.8, 0.75,  0.5, 0.3, 0.2 ]
prag_procentual = 90
valori_sortate = sorted(valori_proprii, reverse=True)
total_varianta = sum(valori_sortate)
suma_cumulata = 0

for i, valoare in enumerate(valori_sortate):
    suma_cumulata += valoare
    procent_explicat = (suma_cumulata / total_varianta) * 100

    if procent_explicat > prag_procentual:
        print(f"Numărul minim de variabile discriminante: {i + 1}")
        break
