# =========================
# #######PANDAS -> CAND PRELUCREZI DATE -> PANDAS
# =========================
# Folosesti asta cand lucrezi cu DataFrame/Series:
# - NaN pe coloane (mean/mode)
# - citire / salvare tabele
# - procente, ponderi
# - shannon, disimilaritate
# - tabele varianta (PCA / FA)
# - metrici clasificare (CM, acurateti, CK)
# - test chi2 pe categorice (dependenta/independenta)
# =========================

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import chi2_contingency


# =========================
# 0) Citire fisiere
# CAND: ai date brute in csv/excel si vrei direct DataFrame
# =========================
def citire_csv(nume_fisier, index_col=None):
    return pd.read_csv(nume_fisier, index_col=index_col)

def citire_excel(nume_fisier, sheet_name=0, index_col=None):
    return pd.read_excel(nume_fisier, sheet_name=sheet_name, index_col=index_col)


# =========================
# 1) NaN in DataFrame (mean/mode)
# CAND: ai lipsuri in coloane; numeric -> mean, categoric -> mode
# =========================
def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


# =========================
# 2) Salvare ndarray ca CSV (cu nume linii/coloane)
# CAND: ai o matrice (ex: R, scoruri, contributii) si vrei tabel csv
# =========================
def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x, nume_linii, nume_coloane)
    if nume_fisier_output is not None:
        temp.to_csv(nume_fisier_output)
    return temp


# =========================
# 3) Salvare DataFrame / Series
# CAND: ai deja tabel/serie si vrei export rapid
# =========================
def salvare_df(t: pd.DataFrame, nume_fisier):
    t.to_csv(nume_fisier)
    return t

def salvare_series(s: pd.Series, nume_fisier):
    s.to_csv(nume_fisier)
    return s


# =========================
# 4) Procente
# CAND: ai o serie de frecvente/totaluri si vrei procente
# =========================
def calcul_procente(t: pd.Series):
    return t * 100 / t.sum()


# =========================
# 5) Pondere
# CAND: ai o serie unde ultima valoare e total (ex: total populatie),
#      iar restul sunt parti; vrei (sum parti) / total
# =========================
def calcul_pondere(t: pd.Series):
    return t.iloc[:-1].sum() / t.iloc[-1]


# =========================
# 6) Medie ponderata
# CAND: ai un DataFrame unde ultima coloana = ponderi (weights)
#      si restul coloanelor = variabile; vrei media ponderata pe coloane
# =========================
def medie_ponderata(t: pd.DataFrame):
    x = t.values
    medii = np.average(x[:, :-1], axis=0, weights=x[:, -1])
    return pd.Series(medii, index=t.columns[:-1])


# =========================
# 7) Shannon (diversitate)
# CAND: ai tabel de numarari pe categorii (coloane) si vrei indice Shannon pe coloane
# =========================
def shannon(t: pd.DataFrame):
    x = t.values
    tx = np.sum(x, axis=0)
    tx[tx == 0] = 1
    p = x / tx
    p[p == 0] = 1
    h = -np.sum(p * np.log2(p), axis=0)
    return pd.Series(h, index=t.columns)


# =========================
# 8) Disimilaritate (Dissimilarity)
# CAND: ai tabel de numarari (linii = zone, coloane = grupe)
#      si vrei disimilaritate pe coloane
# Formula pe coloana j:
# D_j = 0.5 * sum_i | p_ij - p_i |
# p_ij = x_ij / sum_i x_ij
# p_i  = total_i / total_general
# =========================
def calcul_disim(t: pd.DataFrame):
    x = t.values.astype(float)

    total_linie = np.sum(x, axis=1)
    total_general = np.sum(total_linie)
    if total_general == 0:
        return pd.Series([0.0] * t.shape[1], index=t.columns)

    p_i = total_linie / total_general

    disim = []
    for j in range(x.shape[1]):
        col = x[:, j]
        s_col = np.sum(col)
        if s_col == 0:
            disim.append(0.0)
            continue

        p_ij = col / s_col
        d = 0.5 * np.sum(np.abs(p_ij - p_i))
        disim.append(d)

    return pd.Series(disim, index=t.columns)


# =========================
# 9) Standardizare DataFrame (numeric)
# CAND: vrei X_std pt PCA/analize: centrare si optional scalare
# =========================
def standardizare_df(t: pd.DataFrame, scal=True, ddof=0):
    x = t.values.astype(float)
    x_std = x - np.mean(x, axis=0)
    if scal:
        x_std = x_std / np.std(x, axis=0, ddof=ddof)
    return pd.DataFrame(x_std, index=t.index, columns=t.columns)


# =========================
# 10) PCA (pe ndarray)
# CAND: ai matrice numeric (ndarray) si vrei valori/vect proprii + R
# NOTA: asta e "numpy style", dar o tin aici ca e folosita cu tabelele de varianta.
# =========================
def pca(x: np.ndarray, scal=True, ddof=0):
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x, axis=0, ddof=ddof)
    r_v = (1/(n-ddof)) * x_.T @ x_
    valp, vecp = np.linalg.eig(r_v)
    k = np.flip(np.argsort(valp))
    alpa = valp[k]
    a = vecp[:, k]
    return x_, r_v, alpa, a


# =========================
# 11) Tabel varianta (PCA)
# CAND: ai alpha (valori proprii) si vrei varianta + procente
# =========================
def tabelare_varianta(alpha: np.ndarray):
    procent_varianta = alpha * 100 / sum(alpha)
    t = pd.DataFrame(
        data={
            "Varianta": alpha,
            "Varianta cumulata": np.cumsum(alpha),
            "Procent varianta": procent_varianta,
            "Procent cumulat": np.cumsum(procent_varianta)
        },
        index=["C" + str(i) for i in range(1, len(alpha) + 1)]
    )
    t.index.name = "Componenta"
    return t


# =========================
# 12) Tabel varianta (FACT)
# CAND: ai varianta = (valori, procente, procente_cumulate) si vrei tabel
# =========================
def tabelare_varianta_fact(varianta):
    t = pd.DataFrame(
        data={
            "Varianta": varianta[0],
            "Varianta cumulata": np.cumsum(varianta[0]),
            "Procent varianta": varianta[1] * 100,
            "Procent cumulat": varianta[2] * 100
        },
        index=["F" + str(i) for i in range(1, len(varianta[0]) + 1)]
    )
    t.index.name = "Factor"
    return t


# =========================
# 13) Metrici clasificare (CM + acurateti + CK)
# CAND: ai y real, predictie, si lista de clase (labels)
# =========================
def calcul_metrici(y, predictie, clase):
    cm = confusion_matrix(y, predictie, labels=clase)
    t_cm = pd.DataFrame(cm, index=clase, columns=clase)

    t_cm["Acuratete"] = np.diag(cm) * 100 / np.sum(cm, axis=1)

    acuratete_g = sum(np.diag(cm)) * 100 / len(y)
    acuratete_m = t_cm["Acuratete"].mean()
    i_ck = cohen_kappa_score(y, predictie, labels=clase)

    acuratete = pd.Series(
        [acuratete_g, acuratete_m, i_ck],
        index=["Acuratete globala", "Acuratete medie", "Index CK"],
        name="Indicatori acuratete"
    )
    return t_cm, acuratete


# =========================
# 14) Test chi2 (categorice)
# CAND: ai variabile categorice pe coloane si vrei dependenta/independenta
# INPUT: x = ndarray (n x m) cu categorice
# OUTPUT:
#  - teste[j,k] = True daca p-value < alpha (dependenta)
#  - chi2_mat[j,k] = statistica chi2
# =========================
def teste_chi2(x: np.ndarray, alpha=0.01):
    m = x.shape[1]
    teste = np.empty((m, m), dtype=bool)
    chi2_mat = np.empty((m, m), dtype=float)

    for j in range(m):
        for k in range(j, m):
            t = pd.crosstab(x[:, j], x[:, k])
            rez = chi2_contingency(t)
            teste[j, k] = rez[1] < alpha
            chi2_mat[j, k] = rez[0]
            teste[k, j] = teste[j, k]
            chi2_mat[k, j] = chi2_mat[j, k]

    return teste, chi2_mat

