import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from scipy.stats import f

from functii import calcul_metrici, salvare_ndarray
from grafice import f_distributii, f_scatter, show


# ==================================================
# 1. CITIRE DATE + DEFINIRE VARIABILE
# ==================================================

t = pd.read_csv("data_in/Hernia/hernia.csv", index_col=0)

variabile = list(t)
predictori = variabile[:-1]
tinta = variabile[-1]


# ==================================================
# 2. IMPARTIRE TRAIN / TEST
# ==================================================

t_train, t_test, y_train, y_test = train_test_split(
    t[predictori],
    t[tinta],
    test_size=0.2
)


# ==================================================
# 3. EVALUARE PREDICTORI (PUTERE DISCRIMINARE)
# ==================================================

x = t_train[predictori].values
x_ = np.mean(x, axis=0)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(t_train, y_train)

g = model_lda.means_
ponderi = model_lda.priors_
n = len(t_train)
q = len(ponderi)

dg = np.diag(ponderi) * n

sst = (x - x_).T @ (x - x_)
ssb = (g - x_).T @ dg @ (g - x_)
ssw = sst - ssb

f_predictori = (np.diag(ssb) / (q - 1)) / (np.diag(ssw) / (n - q))
pvalues = 1 - f.cdf(f_predictori, q - 1, n - q)

t_predictori = pd.DataFrame(
    {
        "Putere discriminare": f_predictori,
        "PValues": pvalues
    },
    index=predictori
)
t_predictori.round(5).to_csv("data_out/Predictori.csv")


# ==================================================
# 4. DISTRIBUTII PREDICTORI
# ==================================================

clase = model_lda.classes_
for predictor in predictori:
    f_distributii(t_train, predictor, y_train, clase)

show()


# ==================================================
# 5. TESTARE MODEL LDA
# ==================================================

predictie_test_lda = model_lda.predict(t_test)

t_predictii_test = pd.DataFrame(
    {
        tinta: y_test,
        "Predictie LDA": predictie_test_lda
    },
    index=t_test.index
)
t_predictii_test.to_csv("data_out/Predictii_test.csv")

metrici_lda = calcul_metrici(y_test, predictie_test_lda, clase)
metrici_lda[0].to_csv("data_out/CM_lda.csv")
metrici_lda[1].to_csv("data_out/Acuratete_lda.csv")


# ==================================================
# 6. ANALIZA SCORURI DISCRIMINANTE
# ==================================================

z = model_lda.transform(t_train)
etichete_z = ["Z" + str(i + 1) for i in range(q - 1)]

t_z = salvare_ndarray(
    z,
    t_train.index,
    etichete_z,
    "data_out/z.csv"
)

t_gz = t_z.groupby(by=y_train).mean()
gz = t_gz.values

if q > 2:
    for i in range(2, q):
        f_scatter(
            t_z,
            t_gz,
            y_train,
            clase,
            varx="Z1",
            vary="Z" + str(i)
        )

z_ = np.mean(z, axis=0)

sst_z = (z - z_).T @ (z - z_)
ssb_z = (gz - z_).T @ dg @ (gz - z_)
ssw_z = sst_z - ssb_z

f_z = (np.diag(ssb_z) / (q - 1)) / (np.diag(ssw_z) / (n - q))
pvalues_z = 1 - f.cdf(f_z, q - 1, n - q)

t_discriminatori = pd.DataFrame(
    {
        "Putere discriminare": f_z,
        "PValues": pvalues_z
    },
    index=etichete_z
)
t_discriminatori.to_csv("data_out/Discriminatori.csv")

for discriminator in etichete_z:
    f_distributii(
        t_z,
        discriminator,
        y_train,
        clase,
        "Distributii discriminatori"
    )

show()


# ==================================================
# 7. APLICARE MODEL LDA
# ==================================================

t_apply = pd.read_csv(
    "data_in/Hernia/hernia_apply.csv",
    index_col=0
)

predictie_lda = model_lda.predict(t_apply[predictori])

t_predictii = pd.DataFrame(
    {"Predictie LDA": predictie_lda},
    index=t_apply.index
)


# ==================================================
# 8. DISCRIMINARE BAYESIANA
# ==================================================

model_bayes = GaussianNB()
model_bayes.fit(t_train, y_train)

predictie_test_bayes = model_bayes.predict(t_test)
t_predictii_test["Predictie Bayes"] = predictie_test_bayes
t_predictii_test.to_csv("data_out/Predictii_test.csv")

metrici_bayes = calcul_metrici(y_test, predictie_test_bayes, clase)
metrici_bayes[0].to_csv("data_out/CM_bayes.csv")
metrici_bayes[1].to_csv("data_out/Acuratete_bayes.csv")

predictie_bayes = model_bayes.predict(t_apply[predictori])
t_predictii["Predictie Bayes"] = predictie_bayes
t_predictii.to_csv("data_out/Predictii.csv")


# ==================================================
# 9. ANALIZA ERORI
# ==================================================

err_lda = t_predictii_test[y_test != predictie_test_lda]
err_bayes = t_predictii_test[y_test != predictie_test_bayes]
diferente = t_predictii_test[predictie_test_lda != predictie_test_bayes]

err_lda.to_csv("data_out/err_lda.csv")
err_bayes.to_csv("data_out/err_bayes.csv")
diferente.to_csv("data_out/diferente.csv")
