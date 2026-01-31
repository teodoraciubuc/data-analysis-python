#####Grafice
# 🟢 CAND FACI GRAFICE -> functii din seminarii
# Folosesti in functie de cerinta:
# - corelograma: matrice de corelatii
# - plot_varianta: criterii PCA (80%, Kaiser, Cattell)
# - plot_scoruri_corelatii: plot scoruri sau cerc corelatii
# - plot_harta: harta (GeoDataFrame)
# - f_distributii / f_scatter: LDA/discriminare (kde + scatter)
# - plot_ierarhie / plot_partitie / histograme / f_plot_silhouette: clusterizare

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from seaborn import heatmap,kdeplot,scatterplot
from scipy.cluster.hierarchy import dendrogram


# -------------------------------------------------
# Corelograma (din Seminar10_1097/grafice.py)
# CAND: ai matrice R (DataFrame) si vrei heatmap
# -------------------------------------------------
def corelograma(t:pd.DataFrame,vmin=-1,cmap="RdYlBu",annot=True,titlu="Corelograma"):
    f = plt.figure(titlu,figsize=(9,8))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=16,color="b")
    heatmap(t,vmin=vmin,vmax=1,cmap=cmap,annot=annot,ax=ax)
    plt.savefig("graphics/CorelogramaR.png")


# -------------------------------------------------
# Plot varianta (din Seminar10_1097/grafice.py)
# CAND: dupa PCA, ai alpha si vrei k dupa criterii
# -------------------------------------------------
def plot_varianta(alpha:np.ndarray,titlu="Plot varianta componente",
                  x_label="Componenta",procent_minimal_varianta=80,scal=True):
    f = plt.figure(titlu,figsize=(9,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=16,color="b")
    ax.set_xlabel(x_label,fontsize=12)
    ax.set_ylabel("Varianta",fontsize=12)
    m = len(alpha)
    x = np.arange(1,m+1)
    ax.set_xticks(x)
    ax.plot(x,alpha)
    ax.scatter(x,alpha,c="r",alpha=0.5)
    procente_varianta = np.cumsum(alpha)*100/sum(alpha)
    k1 = np.where(procente_varianta>80)[0][0]+1
    ax.axvline(k1,
               c="g",
               label="Criteriul acoperirii minimale ("+str(procent_minimal_varianta)+")")
    k2 = None
    if scal:
        k2 = np.where(alpha>1)[0][-1]+1
        ax.axhline(1,c="m",label="Criteriul Kaiser")
    k3=None
    eps = alpha[:m-1] - alpha[1:]
    sigma = eps[:m-2] - eps[1:]
    negativ = sigma<0
    if any(negativ):
        k3 = np.where(negativ)[0][0]+2
        ax.axvline(k3,c="c",label="Criteriul Cattell")
    ax.legend()
    plt.savefig("graphics/"+titlu+".png")
    return k1,k2,k3


# -------------------------------------------------
# Plot scoruri / cerc corelatii (din Seminar10_1097/grafice.py)
# CAND: plot in planul (C1,C2) pt scoruri (instante) sau corelatii (variabile)
# -------------------------------------------------
def plot_scoruri_corelatii(
        t:pd.DataFrame,
        titlu = "Plot componente",
        x_label="C1",
        y_label="C2",
        etichete=None,
        corelatii=False
        ):
    f = plt.figure(titlu+" "+x_label+" "+y_label,figsize=(9,6))
    ax = f.add_subplot(1,1,1,aspect=1)
    ax.set_title(titlu,fontsize=16,color="b")
    ax.set_xlabel(x_label,fontsize=12)
    ax.set_ylabel(y_label,fontsize=12)
    if corelatii:
        pas = 0.05
        theta = np.arange(0,2*np.pi+pas,pas)
        ax.plot(np.cos(theta),np.sin(theta))
        ax.plot(0.7*np.cos(theta),0.7*np.sin(theta),c="g")
    ax.axvline(0,c="k")
    ax.axhline(0,c="k")
    ax.scatter(t[x_label],t[y_label],c="r",alpha=0.5)
    if etichete is not None:
        n = len(etichete)
        for i in range(n):
            ax.text(t[x_label].iloc[i],t[y_label].iloc[i],etichete[i])
    plt.savefig("graphics/"+titlu+" "+x_label+" "+y_label+".png")


# -------------------------------------------------
# Harta (din Seminar11_1097/grafice.py)
# CAND: ai GeoDataFrame cu geometrie + un tabel cu scoruri pe coduri
# -------------------------------------------------
def plot_harta(gdf:GeoDataFrame,camp_legatura,
               t:pd.DataFrame,camp_harta,cmap="Reds",
               titlu="Harta scoruri"):
    f = plt.figure(titlu + " " + camp_harta, figsize=(9, 6))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu+" "+camp_harta, fontsize=16, color="b")
    gdf_ = gdf[["geometry",camp_legatura]].merge(t,left_on=camp_legatura,right_index=True)
    gdf_.plot(column=camp_harta,cmap=cmap,legend = True,ax = ax)
    plt.savefig("graphics/"+titlu+" "+camp_harta+".png")


# -------------------------------------------------
# Distributii KDE (din Seminar12_1097/grafice.py)
# CAND: compari distributii pe clase (hue=y)
# -------------------------------------------------
def f_distributii(t:pd.DataFrame,variabila,y,clase,titlu="Distributii"):
    f = plt.figure(titlu+" - "+variabila,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu+" - "+variabila,fontdict={"fontsize":16})
    kdeplot(t,x=variabila,hue=y,hue_order=clase,ax=ax,fill=True)


# -------------------------------------------------
# Scatter discriminant (din Seminar12_1097/grafice.py)
# CAND: plotezi instante + centroid/grup (tg) in Z1/Z2 (sau alte axe)
# -------------------------------------------------
def f_scatter(t:pd.DataFrame,tg:pd.DataFrame,y,clase,
              varx="Z1",vary="Z2",titlu="Plot instante in axe discriminante"):
    f = plt.figure(titlu+" - "+varx+" "+vary,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    scatterplot(t,x=varx,y=vary,hue=y,hue_order=clase,ax=ax)
    scatterplot(tg,x=varx,y=vary,hue=clase,hue_order=clase,legend=False,
                ax=ax,marker = "s", s = 100)


# -------------------------------------------------
# Ierarhie (din Seminar13_1097/graphics.py)
# CAND: dupa linkage, vrei dendrograma
# -------------------------------------------------
def plot_ierarhie(h:np.ndarray,etichete=None,color_threshold = 0,titlu="Plot Ierarhie"):
    f = plt.figure(titlu,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    dendrogram(h,color_threshold=color_threshold,labels=etichete,ax=ax)
    if color_threshold!=0:
        ax.axhline(color_threshold,c="r")


# -------------------------------------------------
# Plot partitie (din Seminar13_1097/graphics.py)
# CAND: ai scoruri Z (t_z), centroizi (t_gz) + partitia p si vrei scatter pe clusteri
# -------------------------------------------------
def plot_partitie(
        t_z:pd.DataFrame,
        t_gz:pd.DataFrame,
        p,
        scor_silh,
        titlu,
        etichete=True
):
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu+". Scor Silh:"+str(scor_silh), fontdict={"fontsize": 16})
    clase = np.unique(p)
    scatterplot(t_z,x="Z1",y="Z2",hue=p,hue_order=clase,ax=ax)
    scatterplot(t_gz,x="Z1",y="Z2",hue=clase,hue_order=clase,
                legend=False,marker="s",s = 150, ax=ax)
    if etichete:
        n = len(t_z)
        for i in range(n):
            ax.text(t_z["Z1"].iloc[i],t_z["Z2"].iloc[i],t_z.index[i])


# -------------------------------------------------
# Histograme pe clase (din Seminar13_1097/graphics.py)
# CAND: vrei distributia unei variabile pe fiecare cluster/clasa
# -------------------------------------------------
def histograme(t:pd.DataFrame,variabila,p,titlu="Plot histograme"):
    titlu = titlu+"_"+variabila
    fig = plt.figure(titlu,figsize=(12, 7))
    clase = np.unique(p)
    q = len(clase)
    ax = fig.subplots(1,q,sharey=True)
    fig.suptitle(titlu, fontdict={"fontsize": 16})
    x = t[variabila].values
    for i in range(q):
        axa = ax[i]
        y = x[p==clase[i]]
        assert isinstance(axa,plt.Axes)
        axa.hist(y,10,rwidth=0.9,range=(min(x),max(x)))
        axa.set_xlabel(clase[i])


# -------------------------------------------------
# Silhouette plot (din Seminar13_1097/graphics.py)
# CAND: ai coeficienti silhouette pe instante + media si vrei plot pe clusteri
# -------------------------------------------------
def f_plot_silhouette(partitie, scoruri_silh, scor_silh, titlu="Plot Silhouette"):
    fig = plt.figure(titlu,figsize=(10, 6))
    ax = fig.add_subplot(1,1,1)

    ax.set_title(titlu, fontsize=16)
    clusteri = np.unique(partitie)
    y_lower = 10
    for cluster in clusteri:
        coeficienti = scoruri_silh[partitie == cluster]
        coeficienti.sort()
        size = coeficienti.shape[0]
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, coeficienti,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + size / 2, cluster)
        y_lower = y_upper + 10
    ax.axvline(
        scor_silh,
        color="red",
        linestyle="--",
        label="Coeficient mediu"
    )
    ax.set_xlabel("Coeficienti Silhouette")
    ax.set_ylabel("Cluster")
    ax.legend()


def show():
    plt.show()
