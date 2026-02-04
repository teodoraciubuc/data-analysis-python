import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo, FactorAnalyzer
from sklearn.preprocessing import StandardScaler

df_a=pd.read_csv("a.csv")
df_Coduri=pd.read_csv("Coduri_Localitati.csv")
df_Diversitate=pd.read_csv("DIversitate.csv")

#a1
ani=list(df_Diversitate.columns[2:])
df_Diversitate["Div Medie"] = df_Diversitate[ani].mean(axis=1)
conditie=df_Diversitate["Div Medie"]>df_Diversitate["2021"]
df_final1 = df_Diversitate.loc[conditie, ["Siruta", "Localitate", "Div Medie", "2021"]]
df_final1.to_csv("Cerinta11.csv",index=False)

#a2
df_rezultat=df_Diversitate.merge(df_Coduri[["Siruta","Judet"]],on="Siruta",how="left")
medii=df_rezultat.groupby("Judet")[ani].mean()
maxim=medii.idxmax(axis=1)
df_final2=pd.DataFrame({"Judet":maxim.index,
                        "Anul":maxim.values})
df_final2.to_csv("Cerinta21.csv",index=False)
print("Gata")

#CerintaB
X = df_Diversitate[ani].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# CERINTA B.1: Indexul KMO -> fisierul KMO.csv
kmo_all, kmo_total = calculate_kmo(X_std)
df_kmo = pd.DataFrame({
    'Variabila': ani,
    'KMO': kmo_all
})
df_kmo.loc[len(df_kmo)] = ['KMO_Total', kmo_total]
df_kmo.to_csv("KMO.csv", index=False)


# CERINTA B.2: Scorurile Factoriale -> fisierul f.csv

fa_temp = FactorAnalyzer(rotation=None)
fa_temp.fit(X_std)
ev, _ = fa_temp.get_eigenvalues()
nr_factori = sum(ev > 1)
nr_factori = max(nr_factori, 2)
fa = FactorAnalyzer(n_factors=nr_factori, rotation='varimax')
fa.fit(X_std)
scoruri = fa.transform(X_std)
df_scoruri = pd.DataFrame(scoruri, columns=[f"F{i+1}" for i in range(nr_factori)])
df_scoruri.insert(0, 'Localitate', df_Diversitate['Localitate'])
df_scoruri.to_csv("f.csv", index=False)


# CERINTA B.3: Grafic Scoruri (F1 vs F2)
plt.figure(figsize=(10, 6))
plt.scatter(scoruri[:, 0], scoruri[:, 1], c='blue', alpha=0.6)
tari = df_Diversitate['Localitate'].values
for i in range(min(len(tari), 50)):
    plt.text(scoruri[i, 0], scoruri[i, 1], tari[i], fontsize=9)
plt.title("Plot Scoruri Factoriale (F1 vs F2)")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()

#C
df = pd.read_csv("a.csv")
u1 = df.iloc[:, 0].values
x = np.array([1, 2, -3, 3, 0])
produs = np.dot(x, u1)
lungime_x = np.linalg.norm(x)
cosinus = produs / lungime_x
print("Valoarea Cosinus este:", cosinus)


