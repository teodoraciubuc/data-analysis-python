import sys

import numpy as np
import pandas as pd

from functii import nan_replace, standardizare, salvare_ndarray

pd.set_option("display.max_columns",None)
np.set_printoptions(3,threshold=sys.maxsize,suppress=True)

tabel_date = pd.read_csv("data_in/Teritorial_2022.csv",index_col=0)
# print(tabel_date,type(tabel_date))
variabile_numerice = list(tabel_date)[3:]
x = tabel_date[variabile_numerice].values
print(x,type(x))

nan_replace(x)

x_c = standardizare(x,scal=False)
x_std = standardizare(x)
salvare_ndarray(x_std,tabel_date.index,variabile_numerice,"data_out/x_std.csv")

v = np.cov(x,rowvar=False)
salvare_ndarray(v,variabile_numerice,variabile_numerice,"data_out/V.csv")
r = np.corrcoef(x,rowvar=False)
salvare_ndarray(r,variabile_numerice,variabile_numerice,"data_out/R.csv")

