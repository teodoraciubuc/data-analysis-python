import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from seaborn import scatterplot

def plot_ierarhie(h:np.ndarray,etichete=None,color_threshold = 0,titlu="Plot Ierarhie"):
    f = plt.figure(titlu,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    dendrogram(h,color_threshold=color_threshold,labels=etichete,ax=ax)
    if color_threshold!=0:
        ax.axhline(color_threshold,c="r")

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

def show():
    plt.show()

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
