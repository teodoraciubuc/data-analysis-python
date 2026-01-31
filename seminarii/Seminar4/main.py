import pandas as pd

def calcul_procent(t:pd.Series):
    p = t*100/t.sum()
    return p

def calcul_pondere(t:pd.Series):
    return t.iloc[:-1].sum()/t.iloc[-1]

set_date = pd.read_csv("data_in/IndustriaAlimentara.csv",index_col=0)
industrii = list(set_date)[1:]

# Cerinta 1
angajati_loc = pd.DataFrame(
    set_date[industrii].sum(axis=1),
    columns=["Total_angajati"])
angajati_loc.insert(0,"Localitate",set_date["Localitate"])
cerinta1 = angajati_loc[angajati_loc["Total_angajati"]!=0]
cerinta1.to_csv("data_out/cerinta1.csv")

# Cerinta 2
cerinta2 = set_date[industrii].apply(func=calcul_procent,axis=1)
# print(cerinta2)
cerinta2.to_csv("data_out/cerinta2.csv")

# Cerinta 3
populatie = pd.read_csv(
    "data_in/PopulatieLocalitati.csv",
                        index_col=0)
industrie_alimentara = set_date[industrii].merge(
    populatie[ ["Judet","Populatie"] ],left_index=True,right_index=True
)
industrie_alimentara_jud = industrie_alimentara.groupby(by="Judet").sum()
assert isinstance(industrie_alimentara_jud,pd.DataFrame)
ponderi_judet = industrie_alimentara_jud.apply(func=calcul_pondere,axis=1)
ponderi_judet.name="Pondere"
ponderi_judet.round(3).to_csv("data_out/cerinta3.csv")

