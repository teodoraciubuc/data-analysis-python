import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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



