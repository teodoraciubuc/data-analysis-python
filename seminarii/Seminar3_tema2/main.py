import pandas as pd
pd.set_option("display.max_columns", None)
from functii import nan_replace_df, calcul_procente, calcul_disim, shannon

# Tabel cu populatia pe etnii, la nivel de localitate
t_etnii = pd.read_csv("data_in/Ethnicity.csv", index_col=0)
etnii = list(t_etnii)[1:]

nan_replace_df(t_etnii)


# ==================================================
# 3. LEGARE LOCALITATI -> JUDETE
# ==================================================

# Tabel de corespondenta localitate - judet
coduri_localitati = pd.read_csv(
    "data_in/Coduri_Localitati.csv",
    index_col=0
)

# Adaugare coloana County
t_etnii_loc = t_etnii.merge(
    coduri_localitati,
    left_index=True,
    right_index=True
)

# Agregare pe judet (suma etniilor)
t_etnii_judet = (
    t_etnii_loc[etnii + ["County"]]
    .groupby("County")
    .sum()
)

t_etnii_judet.to_csv("data_out/Ethnicity_County.csv")


# ==================================================
# 4. LEGARE JUDETE -> REGIUNI
# ==================================================

# Tabel de corespondenta judet - regiune
coduri_judete = pd.read_csv(
    "data_in/Coduri_Judete.csv",
    index_col=0
)

# Adaugare coloana Regiune
t_etnii_judet_ = t_etnii_judet.merge(
    coduri_judete,
    left_index=True,
    right_index=True
)

# Agregare pe regiune
t_etnii_regiune = (
    t_etnii_judet_[etnii + ["Regiune"]]
    .groupby("Regiune")
    .sum()
)

t_etnii_regiune.to_csv("data_out/Ethnicity_Region.csv")


# ==================================================
# 5. CALCUL PROCENTE
# ==================================================

# Procente pe localitate
t_etnii_p_loc = t_etnii[etnii].apply(
    func=calcul_procente,
    axis=1
)
t_etnii_p_loc.insert(0, "Localitate", t_etnii["City"])
t_etnii_p_loc.to_csv("data_out/Ethnicity_p.csv")

# Procente pe judet
t_etnii_p_judet = t_etnii_judet.apply(
    func=calcul_procente,
    axis=1
)
t_etnii_p_judet.to_csv("data_out/Ethnicity_p_county.csv")

# Procente pe regiune
t_etnii_p_regiune = t_etnii_regiune.apply(
    func=lambda x: x * 100 / x.sum(),
    axis=1
)
t_etnii_p_regiune.to_csv("data_out/Ethnicity_p_region.csv")


# ==================================================
# 6. DISIMILARITATE PE JUDETE
# ==================================================

# Indicator de disimilaritate pentru fiecare judet
t_disim_judete = (
    t_etnii_loc[etnii + ["County"]]
    .groupby("County")
    .apply(calcul_disim, include_groups=False)
)

t_disim_judete.to_csv("data_out/dissim_county.csv")


# ==================================================
# 7. ENTROPIE (SHANNON) PE JUDETE
# ==================================================

# Indicator de diversitate etnica (entropie Shannon)
t_shannon_judete = (
    t_etnii_loc[etnii + ["County"]]
    .groupby("County")
    .apply(shannon, include_groups=False)
)

t_shannon_judete.to_csv("data_out/shannon_county.csv")
