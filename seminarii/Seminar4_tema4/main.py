import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


def medie_ponderata(t: pd.DataFrame):
    x = t.values
    medii = np.average(x[:, :-1], 0, x[:, -1])
    return pd.Series(medii, t.columns[:-1])


def calcul_procent(t: pd.Series):
    p = t * 100 / t.sum()
    return p


def calcul_pondere(t: pd.Series):
    return t.iloc[:-1].sum() / t.iloc[-1]


set_date = pd.read_csv("data_in/IndustriaAlimentara.csv", index_col=0)
industrii = list(set_date)[1:]

# Cerinta 1
angajati_loc = pd.DataFrame(
    set_date[industrii].sum(axis=1),
    columns=["Total_angajati"])
angajati_loc.insert(0, "Localitate", set_date["Localitate"])
cerinta1 = angajati_loc[angajati_loc["Total_angajati"] != 0]
cerinta1.to_csv("data_out/cerinta1.csv")

# Cerinta 2
cerinta2 = set_date[industrii].apply(func=calcul_procent, axis=1)
# print(cerinta2)
cerinta2.to_csv("data_out/cerinta2.csv")

# Cerinta 3
populatie = pd.read_csv(
    "data_in/PopulatieLocalitati.csv",
    index_col=0)
industrie_alimentara = set_date[industrii].merge(
    populatie[["Judet", "Populatie"]], left_index=True, right_index=True
)
industrie_alimentara_jud = industrie_alimentara.groupby(by="Judet").sum()
assert isinstance(industrie_alimentara_jud, pd.DataFrame)
ponderi_judet = industrie_alimentara_jud.apply(func=calcul_pondere, axis=1)
ponderi_judet.name = "Pondere"
ponderi_judet.round(3).to_csv("data_out/cerinta3.csv")

# Cerinta 4

# print(industrie_alimentara_jud)
cerinta4 = industrie_alimentara_jud[industrii].apply(
    lambda x: x.index[x.argmax()],
    axis=1)
cerinta4.name = "Activitate"
# print(cerinta4)
cerinta4.to_csv("data_out/Cerinta4.csv")

# Cerinta 5
coduri_judete = pd.read_csv("data_in/Coduri_Judete.csv", index_col=0)
# print(industrie_alimentara)
industrie_alimentara_reg = industrie_alimentara[industrii + ["Populatie", "Judet"]].merge(
    coduri_judete["Regiune"], left_on="Judet", right_index=True
)
# print(industrie_alimentara_reg)
cerinta5 = (industrie_alimentara_reg[industrii + ["Populatie", "Regiune"]]. \
    groupby(by="Regiune").apply(
    func=medie_ponderata,
    include_groups=False
))
cerinta5.round(0).to_csv("data_out/Cerinta5.csv")

# Cerinta 6
# print(industrie_alimentara_jud)
totaluri_national = industrie_alimentara_jud.sum(axis=0)
# print(totaluri_national)
cerinta6 = industrie_alimentara_jud.apply(
    lambda x: (x[industrii]/totaluri_national[industrii])/(x["Populatie"]/totaluri_national["Populatie"]) ,
    axis=1)
cerinta6.round(3).to_csv("data_out/Cerinta6.csv")
