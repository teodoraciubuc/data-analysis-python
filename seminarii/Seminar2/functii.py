import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def nan_replace_mode(x:np.ndarray):
    m = x.shape[1]
    for j in range(m):
        y = x[:,j]
        masca = np.array([isinstance(v,float) for v in y])
        v,f = np.unique(y[np.logical_not(masca)],return_counts=True)
        x[masca,j] = v[np.argmax(f)]

def teste_chi2(x:np.ndarray):
    m = x.shape[1]
    teste = np.empty( (m,m),dtype=bool )
    chi2_mat = np.empty( (m,m) )
    for j in range(m):
        for k in range(j,m):
            t = pd.crosstab(x[:,j],x[:,k])
            rez_chi2 = chi2_contingency(t)
            # print(j,k)
            # print(rez_chi2[0],rez_chi2[1])
            teste[j,k] = rez_chi2[1]<0.01
            chi2_mat[j,k] = rez_chi2[0]
            teste[k,j] = teste[j,k]
            chi2_mat[k,j] = chi2_mat[j,k]
    return teste,chi2_mat

def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.to_csv(nume_fisier_output)

