import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from functii import nan_replace_df

pd.set_option("display.max_columns",None)
np.set_printoptions(3,threshold=sys.maxsize,suppress=True)

t = pd.read_csv("data_in/ADN/Y_DNA_Tari.csv",index_col=0)
nan_replace_df(t)

variabile_observate = list(t)[1:]

x = t[variabile_observate].values
x_ = (x - np.mean(x,axis=0))/np.std(x,axis=0)
n,m = x.shape
model_pca = PCA(n_components=m)
model_pca.fit(x_)

alpha = model_pca.explained_variance_*(n-1)/n
c = model_pca.transform(x_)
r_xc = np.corrcoef(x_,c,rowvar=False)[:m,m:]

# Calcule idem main
