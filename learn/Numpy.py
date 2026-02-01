# =========================
# #####NUMPY-> CAND FACI STATISTICA (matrici) -> NUMPY
# =========================
# Folosesti asta cand ai o matrice x (n observatii x m variabile) si vrei:
# - sa tratezi valori lipsa (NaN)
# - sa standardizezi
# - sa faci PCA
# - sa calculezi scoruri (Z)
# - sa calculezi corelatii pe grupe

import numpy as np
import pandas as pd


# =========================
# 1) NaN -> media pe coloana (numeric)
# =========================
def nan_replace(x: np.ndarray):
    # Inlocuieste NaN-urile cu media coloanei (numeric).
    # Modifica x direct (in-place).
    is_nan = np.isnan(x)
    k = np.where(is_nan)
    if k[0].size == 0:
        return
    x[k] = np.nanmean(x[:, k[1]], axis=0)


# =========================
# 2) NaN -> moda pe coloana (categoric)
# =========================
def nan_replace_mode(x: np.ndarray):
    # Pentru coloane categorice (de obicei string) unde lipsurile sunt NaN (float),
    # inlocuieste lipsurile cu moda coloanei.
    # Modifica x direct (in-place).
    m = x.shape[1]
    for j in range(m):
        y = x[:, j]
        masca = np.array([isinstance(v, float) for v in y])  # True pt NaN-uri (float)
        if not np.any(masca):
            continue
        v, f = np.unique(y[np.logical_not(masca)], return_counts=True)
        if v.size == 0:
            continue
        x[masca, j] = v[np.argmax(f)]


# =========================
# 3) Standardizare
# =========================
def standardizare(x: np.ndarray, scal=True, ddof=0):
    # Centreaza pe coloane si optional imparte la abaterea standard.
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x, axis=0, ddof=ddof)
    return x_

# =========================
#  Corelatii pe grupe
# =========================
def calcul_corelatii(x: np.ndarray, y: np.ndarray):
    # Calculeaza matricea de corelatii Pearson pentru fiecare grupa din y.
    # Returneaza dictionar: corelatii[val_grupa] = matrice R (sau None daca grupa e prea mica)
    g = np.unique(y)
    corelatii = {}

    for v in g:
        x_ = x[y == v, :]
        if x_.shape[0] < 2:
            corelatii[v] = None
        else:
            corelatii[v] = np.corrcoef(x_, rowvar=False)

    return corelatii


# =========================
#  PCA
# =========================
def pca(x: np.ndarray, scal=True, ddof=0):
    # Returneaza:
    # x_std (centrata/standardizata), R, valori proprii sortate, vectori proprii sortati
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x_, axis=0, ddof=ddof)

    r_v = (1 / (n - ddof)) * x_.T @ x_
    valp, vecp = np.linalg.eig(r_v)

    k = np.flip(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:, k]
    return x_, r_v, alpha, a

# =========================
# 10) Tabelare varianta (PCA)
# =========================
def tabelare_varianta(alpha: np.ndarray):
    # alpha = valori proprii
    procent_varianta = alpha * 100 / np.sum(alpha)
    t = pd.DataFrame(
        data={
            "Varianta": alpha,
            "Varianta cumulata": np.cumsum(alpha),
            "Procent varianta": procent_varianta,
            "Procent cumulat": np.cumsum(procent_varianta),
        },
        index=["C" + str(i) for i in range(1, len(alpha) + 1)],
    )
    t.index.name = "Componenta"
    return t

# =========================
# Scoruri PCA (Z)
# =========================
def scoruri_pca(x_std: np.ndarray, a: np.ndarray):
    # Z = X_std * A
    return x_std @ a

# =========================
# Calcul partitie np dar cu dataframe
# =========================
def calcul_partitie(h: np.ndarray, k=None):
    # h = linkage matrix (scipy.cluster.hierarchy.linkage)
    # daca k=None -> elbow (alege automat)
    m = h.shape[0]
    n = m + 1

    if k is None:
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        j = n - k

    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2

    c = np.arange(n)
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i
    print("Partitie inainte de codificare:", c)
    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]
    print("Partitie dupa codificare:", partitie)

    return k, color_threshold, np.array(partitie)
