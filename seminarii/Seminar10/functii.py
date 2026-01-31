import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def pca(x: np.ndarray, scal=True, ddof=0):
    n,m = x.shape
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x,axis=0,ddof=ddof)
    r_v = (1/(n-ddof))*x_.T@x_
    valp,vecp = np.linalg.eig(r_v)
    # print(valp)
    # print(vecp)
    k =np.flip(np.argsort(valp))
    # print(k)
    alpa = valp[k]
    a = vecp[:,k]
    return x_,r_v,alpa,a

def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,
                    nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.to_csv(nume_fisier_output)
    return temp

def tabelare_varianta(alpha:np.ndarray):
    procent_varianta = alpha*100/sum(alpha)
    t = pd.DataFrame(
        data={
            "Varianta":alpha,
            "Varianta cumulata":np.cumsum(alpha),
            "Procent varianta":procent_varianta,
            "Procent cumulat":np.cumsum(procent_varianta)
        }, index=["C"+str(i) for i in range(1,len(alpha)+1)]
    )
    t.index.name="Componenta"
    return t

def tabelare_varianta_fact(varianta):
    t = pd.DataFrame(
        data={
            "Varianta": varianta[0],
            "Varianta cumulata": np.cumsum(varianta[0]),
            "Procent varianta": varianta[1]*100,
            "Procent cumulat": varianta[2]*100
        }, index=["F" + str(i) for i in range(1, len(varianta[0]) + 1)]
    )
    t.index.name = "Factor"
    return t