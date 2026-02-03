import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype

#MEREU
df_Fisier=pd.read_csv("Fisier.csv")
def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c:t[c].mean()}, inplace=True)
            else:
                t.fillna({c:t[c].mode()[0]}, inplace=True)
nan_replace_df(df_Fisier)

# region ~analiza factoriala cu 5 variabile si 2 factori comuni(s au format factorloadings).Sa se afiseze factorii cu cea mai mare varianta
comunalitati = (df_Fisier ** 2).sum(axis=1)
psi = 1 - comunalitati
print("Raspunsul e:", int(np.argmax(psi.values)+1))
#endregion
#region ~analiza factoriala cu 5 variabile (s au format coeficienti corelatie in 2 fisiere diferite-unii normali nii de corelatie partiala). Se cere KMO global
R = pd.read_csv('Fisier_matrice_corelatie.csv', index_col=0).values
P = pd.read_csv('Fisier_matrice_cor_partiala.csv', index_col=0).values

R2 = R ** 2
P2 = P ** 2
mask = np.ones(R2.shape) - np.eye(R2.shape[0])

numitorul = np.sum(R2 * mask)
numitorul_plus_partial = numitorul + np.sum(P2 * mask)

kmo_global = numitorul / numitorul_plus_partial
print(f"Indexul KMO global este: {kmo_global}")
#endregion
#region ~analiza discriminanta liniara aplicata pe ... instante descrise prin .. variabile s-au obtinut valorile proprii [... , ..., ...].Sa se afiseze nr min de variabile discriminante pt a acoperii peste 90% din putere
valori_proprii = [... , ... , ... ] #inclouiesti din cerinta
prag_procentual = 90 #schimb in functie de ce scrie in cerinta

#restu copy
valori_sortate = sorted(valori_proprii, reverse=True)
total_varianta = sum(valori_sortate)
suma_cumulata = 0

for i, valoare in enumerate(valori_sortate):
    suma_cumulata += valoare
    procent_explicat = (suma_cumulata / total_varianta) * 100

    if procent_explicat > prag_procentual:
        print(f"Numărul minim de variabile discriminante: {i + 1}")
        break
#endregion
#region ~avem matricea vectorilor pe coloane si valori proprii (..., .... , ....). Sa se afiseze scorurile din instanta x=[....,...,...]
file_name = 'dataIN/a.csv'
x = np.array([..,..])  #instanta
#copy
df_a = pd.read_csv(file_name, header=None)
A = df_a.values
scoruri = np.dot(x, A)
print("Scorurile obținute de instanța x sunt:",scoruri)
#endregion
#region ~criteriul cattel = sunt salvate instante pt componente principale standardizata pe sase variabile obs. sa se afiseze nr de componente principale semnificative criteriului cattel
ev = np.var(pd.read_csv('df_fisier.csv', index_col=0), axis=0)
diffs = np.diff(ev)
print(np.argmin(diffs) + 1)
#endregion
#region ~analiza componente principlae cu .. variabile. Valorile proprii sunt: (...). Să se afișeze la consolă variabilele observate pentru care primele două componente principale explică în comun mai mult de 90% din varianță (comunalitatea). Se va afișa indexul variabilei (număr de la 1 la 5) sau eticheta variabilei (Xk, unde k poate fi de la 1 la 5).
# valori_proprii = [3.019, 1.2203, 0.6536, 0.102, 0.005]
# prag_varianta = 0.90
# k_componente = 2 #cate componente se cer
file_read = 'dataIN/a.csv'

#copy
A = pd.read_csv(file_read, header=None).values
L = A * np.sqrt(valori_proprii)
comunalitati_k = np.sum(L[:, :k_componente]**2, axis=1)
print(f"Variabilele care explică peste {prag_varianta*100}% din varianță prin primele {k_componente} CP:")

for i, val in enumerate(comunalitati_k):
    if val > prag_varianta:
        # Afișăm indexul (i+1) și eticheta X(i+1)
        print(f"X{i+1} (Index: {i+1}) -> Comunalitate: {val:.4f}")
#endregion
#region criteriul kaiser = in fisier sunt salvate comunalitatile pt un model de pca standardizata cu 5 var obs. sa se afiseze nr de componente semnificaitve kaiser
df_comm = pd.read_csv('comm.csv', index_col=0)

# Dacă fișierul conține direct valorile proprii pe coloane:
valori_proprii = df_comm.values.flatten()

# Dacă fișierul conține scoruri/comunalități și trebuie calculată varianța:
# valori_proprii = np.var(df_comm, axis=0)

componente_semnificative = valori_proprii[valori_proprii > 1]
numar_componente = len(componente_semnificative)
print(f"Numărul de componente principale semnificative (Kaiser): {numar_componente}")
#endregion
#region Într-un model de analiză discriminantă liniară cu 6 variabile predictor și 4 clase, matricea de covarianță totală (T) a fost salvată în fișierul T.csv iar centrii de clase au fost salvați în fișierul G.csv. Predictorii sunt X1, ..., X6 iar clasele sunt etichetate cu A, B, C și D. Să se afișeze la consolă clasa in care va fi clasificată instanța x = [118, 19.8, 2.5, 8.3, 61.1, 8.3].Indicație. Inversa matricei de covarianță totală va fi calculată prin funcția np.linalg.inv(), unde np este numele de import al pachetului numpy. (1 punct)
import pandas as pd
import numpy as np

# 1. DATELE DIN CERINȚĂ (Modifici aici valorile și numele fișierelor)
fisier_T = 'T.csv'  # Matricea de covarianță totală
fisier_G = 'G.csv'  # Centrii claselor (Media predictorilor pe fiecare clasă)
x = np.array([118, 19.8, 2.5, 8.3, 61.1, 8.3]) # Instanța de clasificat

T = pd.read_csv(fisier_T, index_col=0).values
G = pd.read_csv(fisier_G, index_col=0)
nume_clase = G.index
centri = G.values
T_inv = np.linalg.inv(T)
distante = []
for i in range(len(centri)):
    dif = x - centri[i]
    d2 = dif @ T_inv @ dif.T
    distante.append(d2)

index_minim = np.argmin(distante)
clasa_predictie = nume_clase[index_minim]

print(f"Instanța x este clasificată în clasa: {clasa_predictie}")
#endregion