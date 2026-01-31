import sys
import numpy as np
import pandas as pd

# Functii pentru PCA si salvare rezultate
from functii import nan_replace_df, pca, salvare_ndarray, tabelare_varianta

# Functii pentru grafice
from grafice import corelograma, show, plot_varianta

pd.set_option("display.max_columns", None)
np.set_printoptions(3, threshold=sys.maxsize, suppress=True)

# Tabel cu datele genetice pe tari
t = pd.read_csv("data_in/ADN/Y_DNA_Tari.csv", index_col=0)

# Inlocuire valori lipsa (numeric -> media, text -> moda)
nan_replace_df(t)


# ==================================================
# 2. SELECTARE VARIABILE OBSERVATE
# ==================================================

# Coloanele numerice (indicatorii genetici)
variabile_observate = list(t)[1:]

# Trecere la numpy (statistica)
x = t[variabile_observate].values


# ==================================================
# 3. PCA
# ==================================================

# pca intoarce:
# x_     -> date standardizate
# r_v    -> matricea de corelatie intre variabile
# alpha  -> valori proprii (varianta)
# a      -> vectori proprii (axe componente)
x_, r_v, alpha, a = pca(x)


# ==================================================
# 4. ANALIZA VARIANTEI
# ==================================================

# Salvare matrice de corelatie
t_r_v = salvare_ndarray(
    r_v,
    variabile_observate,
    variabile_observate,
    "data_out/R.csv"
)

# Afisare corelograma (grafic corelatii)
corelograma(
    t_r_v,
    titlu="Corelograma variabile observate"
)

# Tabel cu varianta explicata de fiecare componenta
t_varianta = tabelare_varianta(alpha)

# Salvare tabel varianta
t_varianta.round(3).to_csv("data_out/Varianta.csv")

# Grafic scree plot (cate componente pastram)
k = plot_varianta(alpha)


# ==================================================
# 5. CORELATII VARIABILE – COMPONENTE
# ==================================================

# Calcul componente principale (proiectia datelor)
c = x_ @ a

# Etichete pentru componente (C1, C2, ...)
etichete_componente = list(t_varianta.index)

# Salvare componente
t_c = salvare_ndarray(
    c,
    t.index,
    etichete_componente,
    "data_out/C.csv"
)

# Corelatii dintre variabilele observate si componente
n, m = x.shape
r_xc = np.corrcoef(x_, c, rowvar=False)[:m, m:]

# Salvare corelatii factoriale
t_r_xc = salvare_ndarray(
    r_xc,
    variabile_observate,
    etichete_componente,
    "data_out/r_xc.csv"
)

# Afisare corelograma factoriala
corelograma(
    t_r_xc,
    titlu="Corelatii factoriale"
)


# ==================================================
# 6. ANALIZA SCORURILOR
# ==================================================

# Scoruri standardizate ale componentelor
s = c / np.sqrt(alpha)

# Salvare scoruri
t_s = salvare_ndarray(
    s,
    t.index,
    etichete_componente,
    "data_out/s.csv"
)

# Afiseaza toate graficele generate
show()
