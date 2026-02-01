import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_Country=pd.read_csv("dataIN/CountryContinents.csv",index_col=0)
df_g20=pd.read_csv("dataIN/g20.csv",index_col=0)
df_GlobalInd=pd.read_csv("dataIN/GlobalIndicatorsPerCapita_2021.csv",index_col=0)

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(df_Country)
nan_replace_df(df_g20)
nan_replace_df(df_GlobalInd)

# # ------------CERINTA A.1.-------------
indicatori_valori_cerinta1 = df_GlobalInd.iloc[:,8:]
cerinta1= df_GlobalInd[['Country']].copy()
cerinta1['Valoare Adaugata'] = indicatori_valori_cerinta1.sum(axis=1)
cerinta1.to_csv('./dataOUT/Cerinta1.csv', index=True)
# # --------------------------------------------------------------

# CERINTA A2

indicatori2=df_GlobalInd.iloc[:,1:]
df_final=df_Country.merge(indicatori2,left_index=True,right_index=True)
coeficientVariatie=df_final.groupby('Continent')[indicatori2.columns].agg(lambda x: x.std(ddof=1)/x.mean())
coeficientVariatie.reset_index().to_csv('./dataOUT/coeficientVariatie.csv', index=True)

#Cerinta B1

varianteComponentePrincipale=df_GlobalInd.iloc[:,1:].values
varianteS=StandardScaler().fit_transform(varianteComponentePrincipale)
pca=PCA()
c=pca.fit_transform(varianteS)
n,m=varianteS.shape
variatii=pca.explained_variance_
alpha=variatii*(n-1)/n
print(alpha)
print('Variatii:',variatii)

#Cerinta B2

scoruri = pd.DataFrame(c, index= df_GlobalInd.index, columns=[f"CP{i}" for i in range(1, c.shape[1]+1)])
scoruri.to_csv('./dataOUT/scoruri.csv', index=True)

#Cerinta B3
plt.figure(figsize=(8,6))
plt.scatter(c[:,0], c[:,1])

plt.xlabel("CP1")
plt.ylabel("CP2")
plt.title("Scoruri in primele 2 axe principale (CP1,CP2)")
plt.axhline(0)
plt.axvline(0)

plt.show()

#Cerinta C
comunalitati=(df_g20 ** 2).sum(axis=1)
psi=1-comunalitati
print("Raspunsul e:", int(np.argmax(psi.values)+1))

