import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap


def corelograma(t:pd.DataFrame,vmin=-1,cmap="RdYlBu",annot=True,titlu="Corelograma"):
    f = plt.figure(titlu,figsize=(9,8))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=16,color="b")
    heatmap(t,vmin=vmin,vmax=1,cmap=cmap,annot=annot,ax=ax)
    plt.savefig("graphics/CorelogramaR.png")

def show():
    plt.show()

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
