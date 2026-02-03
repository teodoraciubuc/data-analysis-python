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
coloane=list(df_Diversitate.columns[2:])
conditie=(df_Diversitate[coloane]==0).any(axis=1)
df_final=df_Diversitate[conditie]
df_final.to_csv("Cerinta1.csv",index=False)
print("Gata")


#a2
#merge+groupby in functie de care are mai multe din col
#avem conditie
#unesti merge [conditie, coloane]
df=df_Diversitate.merge(df_Coduri[["Siruta","Judet"]], on='Siruta')
df["DiversitateMaxima"]=df[coloane].mean(axis=1)
maxim=df.groupby("Judet")["DiversitateMaxima"].idxmax()
df_rezultat=df.loc[maxim,["Judet","Localitate","DiversitateMaxima"]]
df_rezultat.to_csv("Cerinta2.csv",index=False)
print("Gata")
