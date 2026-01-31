import pandas as pd

from functii import nan_replace_mode, teste_chi2, salvare_ndarray

pd.set_option("display.max_columns",None)

set_date = pd.read_csv("data_in/RSC.csv",dtype={"Parteneri":str,"Perioada":str})
# print(set_date.dtypes)
# print(set_date)

variabile_categoriale = list(set_date.columns[1:])

x = set_date[variabile_categoriale].values
# print(x,type(x[0,0]))

nan_replace_mode(x)
# print(x)
teste,chi2_mat = teste_chi2(x)
salvare_ndarray(teste,
                variabile_categoriale,
                variabile_categoriale,
                "data_out/teste.csv"
                )
salvare_ndarray(chi2_mat,
                variabile_categoriale,
                variabile_categoriale,
                "data_out/chi2.csv"
                )
