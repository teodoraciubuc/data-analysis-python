import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_a=pd.read_csv("a.csv")
df_Coduri=pd.read_csv("Coduri_Localitati.csv")
df_Diversitate=pd.read_csv("DIversitate.csv")

#a1
#faci coloane/indicatori/ani
#conditie
#faci df_final1 si afisezi
ani=list(df_Diversitate.columns[2:])
media=df_Diversitate[ani].mean(axis=1)
conditie=media>df_Diversitate["2021"]
df_final1=df_Diversitate[conditie]
df_final1.to_csv("Cerinta1.csv",index=False)
print("Gata")

#a2
#merge+groupby in functie de care are mai multe din col
#avem conditie
# #unesti merge [conditie, coloane]
df_rezultat=df_Diversitate.merge(df_Coduri[["Siruta","Judet"]],on="Siruta",how="left")
mediiJudet=df_rezultat.groupby("Judet")[ani].mean()
maxim=mediiJudet.idxmax(axis=1)
df_final2=maxim.reset_index()
df_final2.columns=["Judet","Anul"]
df_final2.to_csv("Cerinta2.csv",index=False)
print("Gata")
