import sys
import numpy as np
import pandas as pd

# Functii PCA si prelucrare rezultate
from functii import nan_replace_df, pca, salvare_ndarray, tabelare_varianta

# Functii pentru grafice
from grafice import (
    corelograma,
    show,
    plot_varianta,
    plot_scoruri_corelatii
)

# ==================================================
# 0. SETARI AFISARE
# ==================================================

pd.set_option("display.max_columns", None)
np.set_printoptions(3, threshold=sys.maxsize, suppress=True)


# ==================================================
# 1. CITIRE DATE + CURATARE
# ==================================================

# Date genetice pe tari
t = pd.read_csv("data_in/ADN/Y_DNA_Tari.csv", index_col=0)

# Inlocuire valori lipsa
nan_replace_df(t)


# ==================================================
# 2. SELECTARE VARIABILE OBSERVATE
# ==================================================

variabile_observate = list(t)[1:]
x = t[variabile_observate].values


# ==================================================
# 3. PCA
# ==================================================

# x_     -> date standardizate
# r_v    -> matrice corelatii variabile
# alpha  -> valori proprii (varianta)
# a      -> vectori proprii
x_, r_v, alpha, a = pca(x)


# ==================================================
# 4. ANALIZA VARIANTEI
# ==================================================

# Salvare corelatii intre variabile
t_r_v = salvare_ndarray(
    r_v,
    variabile_observate,
    variabile_observate,
    "data_out/R.csv"
)

# Corelograma variabile
corelograma(t_r_v, titlu="Corelograma variabile observate")

# Tabel varianta explicata
t_varianta = tabelare_varianta(alpha)
t_varianta.round(3).to_csv("data_out/Varianta.csv")

# Scree plot + selectie nr optim de componente
k = plot_varianta(alpha)
k_optim = min(v for v in k if v is not None)


# ==================================================
# 5. CORELATII VARIABILE – COMPONENTE
# ==================================================

# Calcul componente principale
c = x_ @ a
etichete_componente = list(t_varianta.index)

# Salvare componente
t_c = salvare_ndarray(
    c,
    t.index,
    etichete_componente,
    "data_out/C.csv"
)

# Corelatii variabile – componente
n, m = x.shape
r_xc = np.corrcoef(x_, c, rowvar=False)[:m, m:]

t_r_xc = salvare_ndarray(
    r_xc,
    variabile_observate,
    etichete_componente,
    "data_out/r_xc.csv"
)

# Corelograma corelatii factoriale
corelograma(t_r_xc, titlu="Corelatii factoriale")

# Cercurile corelatiilor
for i in range(2, k_optim + 1):
    plot_scoruri_corelatii(
        t_r_xc,
        titlu="Plot corelatii",
        y_label="C" + str(i),
        etichete=t_r_xc.index if m < 50 else None,
        corelatii=True
    )


# ==================================================
# 6. ANALIZA SCORURILOR
# ==================================================

# Scoruri standardizate
s = c / np.sqrt(alpha)

t_s = salvare_ndarray(
    s,
    t.index,
    etichete_componente,
    "data_out/s.csv"
)

# Grafice scoruri
for i in range(2, k_optim + 1):
    plot_scoruri_corelatii(
        t_c,
        y_label="C" + str(i),
        etichete=t.index if n < 50 else None
    )


# ==================================================
# 7. METRICI PCA
# ==================================================

# Cosinusuri patrate (calitatea reprezentarii observatiilor)
c2 = c * c
cosin2 = (c2.T / np.sum(c2, axis=1)).T

t_cosin2 = salvare_ndarray(
    cosin2,
    t.index,
    etichete_componente,
    "data_out/Cosin.csv"
)

if n < 50:
    corelograma(
        t_cosin2,
        prag=0,
        cmap="Reds",
        annot=False,
        titlu="Valori cosinus"
    )


# ==================================================
# 8. CONTRIBUTII + COMUNALITATI
# ==================================================

# Contributii (procente)
contrib = c2 * 100 / np.sum(c2, axis=0)

t_contrib = salvare_ndarray(
    contrib,
    t.index,
    etichete_componente,
    "data_out/Contrib.csv"
)

# Comunalitati
r2 = r_xc * r_xc
comm = np.cumsum(r2, axis=1)

t_comm = salvare_ndarray(
    comm,
    variabile_observate,
    etichete_componente,
    "data_out/Comm.csv"
)

corelograma(
    t_comm,
    prag=0,
    cmap="Blues",
    annot=False,
    titlu="Comunalitati"
)


# ==================================================
# 9. AFISARE GRAFICE
# ==================================================

show()
