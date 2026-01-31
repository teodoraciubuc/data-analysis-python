import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from functii import nan_replace_df, calcul_partitie, salvare_ndarray
from graphics import (
    plot_ierarhie,
    plot_partitie,
    histograme,
    f_plot_silhouette,
    show
)

t = pd.read_csv(
    "data_in/MortalitateRO2019/mortalitate_ro.csv",
    index_col=1
)
nan_replace_df(t)

# ==================================================
# 2. SELECTARE VARIABILE OBSERVATE
# ==================================================

variabile_observate = list(t)[1:]
x = t[variabile_observate].values


# ==================================================
# 3. CLUSTERIZARE IERARHICA
# ==================================================

metoda_grupare = "complete"

# Matrice de ierarhie
h = linkage(x, metoda_grupare)

# Dendrograma
plot_ierarhie(h, t.index)
show()


# ==================================================
# 4. PARTITIE OPTIMALA
# ==================================================

k_opt, color_threshold_opt, p_opt = calcul_partitie(h)

plot_ierarhie(
    h,
    t.index,
    color_threshold_opt,
    "Partitia optimala"
)

t_partitii = pd.DataFrame(
    {"Partitie O": p_opt},
    index=t.index
)


# ==================================================
# 5. EVALUARE PARTITIE OPTIMALA (SILHOUETTE)
# ==================================================

scoruri_silh_opt = silhouette_samples(x, p_opt)
scor_silh_opt = silhouette_score(x, p_opt)

t_partitii["Scor_Silh_Opt"] = scoruri_silh_opt

f_plot_silhouette(
    p_opt,
    scoruri_silh_opt,
    scor_silh_opt,
    "Scoruri Silhouette - Partitia optimala"
)


# ==================================================
# 6. VIZUALIZARE PARTITIE OPTIMALA (PCA)
# ==================================================

acp = PCA(2)
z = acp.fit_transform(x)

t_z = salvare_ndarray(
    z,
    t.index,
    ["Z1", "Z2"],
    None
)

tg_z = t_z.groupby(by=p_opt).mean()

plot_partitie(
    t_z,
    tg_z,
    p_opt,
    scor_silh_opt,
    "Partitia optimala"
)

# Histograme pe variabile
for variabila in variabile_observate:
    histograme(
        t,
        variabila,
        p_opt,
        "Histograme partitia optimala"
    )

show()


# ==================================================
# 7. PARTITIE CU NUMAR FIX DE CLUSTERI (k = 3)
# ==================================================

k = 3
k_fix, color_threshold_k, p_k = calcul_partitie(h, k)

plot_ierarhie(
    h,
    t.index,
    color_threshold_k,
    "Partitia din " + str(k) + " clusteri"
)

t_partitii["Partitie " + str(k)] = p_k


# ==================================================
# 8. EVALUARE PARTITIE k (SILHOUETTE)
# ==================================================

scoruri_silh_k = silhouette_samples(x, p_k)
scor_silh_k = silhouette_score(x, p_k)

t_partitii["Scor_Silh_" + str(k)] = scoruri_silh_k

f_plot_silhouette(
    p_k,
    scoruri_silh_k,
    scor_silh_k,
    "Scoruri Silhouette - Partitia din " + str(k) + " clusteri"
)


# ==================================================
# 9. VIZUALIZARE PARTITIE k (PCA)
# ==================================================

tg_z = t_z.groupby(by=p_k).mean()

plot_partitie(
    t_z,
    tg_z,
    p_k,
    scor_silh_k,
    "Partitia din " + str(k) + " clusteri"
)

# Histograme pe variabile
for variabila in variabile_observate:
    histograme(
        t,
        variabila,
        p_k,
        "Histograme partitia din " + str(k) + " clusteri"
    )

show()

t_partitii.to_csv("data_out/Partitii.csv")
