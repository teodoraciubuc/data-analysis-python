import sys
import numpy as np
import pandas as pd
import factor_analyzer as fa

from functii import nan_replace_df, salvare_ndarray, tabelare_varianta_fact
from grafice import corelograma, show, plot_varianta
pd.set_option("display.max_columns", None)
np.set_printoptions(3, threshold=sys.maxsize, suppress=True)

# Date de mortalitate (pe judete / ani)
t = pd.read_csv(
    "data_in/MortalitateRO2019/mortalitate_ro.csv",
    index_col=1
)

# Inlocuire valori lipsa
nan_replace_df(t)

# ==================================================
# 2. SELECTARE VARIABILE OBSERVATE
# ==================================================

variabile_observate = list(t)[1:]
x = t[variabile_observate].values
n, m = x.shape


# ==================================================
# 3. TESTE DE FACTORABILITATE
# ==================================================

# Test Bartlett (daca variabilele sunt corelate)
test_bartlett = fa.calculate_bartlett_sphericity(x)

if test_bartlett[1] > 0.01:
    print("Lipsa factorabilitate!")
    exit(0)

# Test KMO (adecvarea esantionului)
kmo = fa.calculate_kmo(x)

t_kmo = pd.DataFrame(
    data={"KMO": np.append(kmo[0], kmo[1])},
    index=variabile_observate + ["Global"]
)
t_kmo.to_csv("data_out_fa/kmo.csv")

corelograma(
    t_kmo,
    prag=0,
    cmap="Reds",
    titlu="Index KMO"
)


# ==================================================
# 4. MATRICEA DE CORELATII
# ==================================================

r = np.corrcoef(x, rowvar=False)

t_r = salvare_ndarray(
    r,
    variabile_observate,
    variabile_observate,
    "data_out_fa/R.csv"
)

corelograma(t_r, annot=m < 15)


# ==================================================
# 5. CONSTRUIRE MODEL FACTORIAL
# ==================================================

# Model fara rotatie, cu numar maxim de factori
model_fact = fa.FactorAnalyzer(
    n_factors=m,
    rotation=None
)

model_fact.fit(x)


# ==================================================
# 6. ANALIZA VARIANTEI FACTORIALE
# ==================================================

# Varianta explicata de factori
varianta = model_fact.get_factor_variance()
alpha = varianta[0]

t_varianta = tabelare_varianta_fact(varianta)
t_varianta.to_csv("data_out_fa/Varianta.csv")

# Scree plot + selectie numar optim de factori
k = plot_varianta(
    alpha,
    titlu="Plot varianta factori",
    eticheta="Factor",
    prag=70
)

nr_factori = min(v for v in k if v is not None)


# ==================================================
# 7. CORELATII VARIABILE – FACTORI
# ==================================================

# Incarcari factoriale (doar factorii pastrati)
r_xf = model_fact.loadings_[:, :nr_factori]

etichete_factori = list(t_varianta.index)

t_r_xf = salvare_ndarray(
    r_xf,
    variabile_observate,
    etichete_factori[:nr_factori],
    "data_out_fa/r_xf.csv"
)

corelograma(
    t_r_xf,
    titlu="Corelatii factoriale"
)

show()
