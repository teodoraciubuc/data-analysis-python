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
#a1-coeficient variatie
cv = df_Air[indicatori].std(ddof=1) / df_Air[indicatori].mean()
df_final = cv.reset_index()
df_final.columns=["Indicator","CV"]
df_final.to_csv("AirQuality.csv",index=False)

print("Gata")

#a2
#merge+groupby in functie de care are mai multe din col
#avem conditie
#unesti merge [conditie, coloane]

df_rezultat=df_Air.merge(df_Coduri[["CountryID","Continent"]],left_on="CountryId",right_on="CountryID",how="left")

mean_cont = df_rezultat.groupby("Continent")[indicatori].mean()
std_cont  = df_rezultat.groupby("Continent")[indicatori].std(ddof=1)
cv_cont = std_cont / mean_cont.replace(0, np.nan)

df_final2 = pd.DataFrame({
    "Continent": cv_cont.index,
    "Indicator": cv_cont.idxmax(axis=1).values,
    "CV": cv_cont.max(axis=1).values
})
df_final2.to_csv("Cerinta2.csv",index=False)
print("Gata")



