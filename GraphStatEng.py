# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:40:53 2023

@author: Rodrigo lopez Aburto
"""

## Funciones miscellaneas para el NoteBook de clasificacion medinate el algoritmo
## de Sel-Organizing Maps (SOM)
## RLA 

## Bibliotecas utilizadas
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

#####################################################################################################################################

def Estadigrafos (Data, names):
    """
    Funcion para generar las tablas de estadigrafos usadas en la seccion del 
    analisis exploratorio de datos. La funcion recibe un DataFrame y una lista
    con los mnemonicos para calcular los estadigrafos
    
    Parameters
    ----------
    Data : pd.DataFrame
        Datos a los cuales se les estimaran los estadigrafos.
    names : list of str
        Nombres de las columnas de los datos a las que se les estimaran los estadigrafos.
        Los nombres de la lista deben coincidir con los nombres de las columnas del DataFrame.

    Returns
    -------
    pd.DataFrame con los estadigrafos (Muestras, Minimo, 1er Cuartil, Mediana, Media,
    3er Cuartil, Maximo, Rango, Rango Intercuartil, Varianza, Desviacion Estandar, Simetria y Curtosis)
    estimados para cada columna dentro de la lista de nombres.
    """
   
    Encabezados = ['Samples', 'Minimum', '1st quartile', 'Median', 'Mean',
                   '3rd quartile', 'Maximum', 'Range', 'IQR', 
                   'Variance', 'Std dev', 'Skewness', 'Kurtosis']
    Resultado = pd.DataFrame(data=np.zeros((len(Encabezados),len(names))), columns=names, index=Encabezados)
    for name in names:
        Muestras = int(Data[name].shape[0])
        Minimo = Data[name].min()
        Cuartil_1 = Data[name].quantile([0.25])[0.25]
        Mediana = Data[name].quantile([0.5])[0.5]
        Media = Data[name].mean()
        Cuartil_3 = Data[name].quantile([0.75])[0.75]
        Maximo = Data[name].max()
        Rango = Maximo - Minimo
        IQR = Cuartil_3 - Cuartil_1
        Varianza = Data[name].var()
        Desv_Estandar = Data[name].std()
        Simetria = Data[name].skew()
        Curtosis= Data[name].kurtosis()
        Resultado[name] = [Muestras, Minimo, Cuartil_1, Mediana, Media, Cuartil_3, Maximo,
                     Rango, IQR, Varianza, Desv_Estandar, Simetria, Curtosis]
    return(Resultado)


####################################################################################################################################

def Contar_Clases(Datos, Etiqueta):
    """
    Funcion para contar las clases de un pd.DataFrame

    Parameters
    ----------
    Datos : pd.DataFrame
        Datos a los cuales se les contara alguna variable categorica.
    Etiqueta : str
        Nombre de la columna donde se encuentran los datos categoricos.

    Returns
    -------
    pd.DataFrame con el conteo absoluto y en porcentaje.

    """
    if Etiqueta in Datos:
        Categories_original = pd.value_counts(Datos[Etiqueta])
        df_Categories_original = pd.concat([Categories_original, Categories_original], axis=1)
        df_Categories_original.reset_index(inplace=True)
        df_Categories_original.columns = [Etiqueta, 'COUNT', 'PERCENT']
        for i in range(len(Categories_original)):
            df_Categories_original.loc[i, 'PERCENT'] = (df_Categories_original.loc[i, 'COUNT'] * 100.0 / len(Datos))
        df_Categories_original.sort_values(by=[Etiqueta], inplace=True)
        df_Categories_original.reset_index(inplace=True); df_Categories_original.drop(['index'], axis=1, inplace=True)
        return(df_Categories_original)
    else:
        Categories_Set = pd.value_counts(Datos)
        df_Categories_Set = pd.concat([Categories_Set, Categories_Set], axis=1)
        df_Categories_Set.reset_index(inplace=True)
        df_Categories_Set.columns = [Etiqueta, 'COUNT', 'PERCENT']
        for i in range(len(Categories_Set)):
            df_Categories_Set.loc[i, 'PERCENT'] = (df_Categories_Set.loc[i, 'COUNT'] * 100.0 / len(Datos))
        df_Categories_Set.sort_values(by=[Etiqueta], inplace=True)
        df_Categories_Set.reset_index(inplace=True); df_Categories_Set.drop(['index'], axis=1, inplace=True)
        return(df_Categories_Set)
    
    
####################################################################################################################################

def Upscale(Data,continuos,window_len,categorical='none',ref='DEPTH'):
    """
    Funcion para escalar datos.

    Parameters
    ----------
    Data : TYPE
        DESCRIPTION.
    continuous : TYPE
        DESCRIPTION.
    categorical : TYPE
        DESCRIPTION.
    window_len : TYPE
        DESCRIPTION.
    ref : TYPE, optional
        DESCRIPTION. The default is 'DEPTH'.

    Returns
    -------
    None.

    """
    temp_index = []
    for x in range(round(len(Data)/window_len)):
        temp_index.append((math.floor(window_len/2)) + (window_len * x))
    df_indexed = pd.DataFrame(index=temp_index)
    if categorical == 'none':
        df_continuos = Data[continuos].rolling(window_len, center=True).median()
        df_upscaled = Data[ref].filter(items=temp_index,axis=0).to_frame().join(df_continuos.filter(items=temp_index,axis=0))
    else:
        df_categorical = Data[categorical].rolling(window_len, center=True).apply(lambda x: mode(x)[0])
        df_upscaled1 = Data[ref].filter(items=temp_index,axis=0).to_frame().join(df_continuos.filter(items=temp_index,axis=0))
        df_upscaled = df_upscaled1.join(df_temp1.filter(items=temp_index,axis=0))
    return(df_upscaled)
    
####################################################################################################################################

def Tabla_PCA(modelo):
    """
    Funcion para estimar la contribucion a la varianza de cada una de las componentes principales de un modelo

    Parameters
    ----------
    modelo :sklearn.model
        Modelo ajustado (tipo de dato propio de sklearn).

    Returns
    -------
    pd.DataFrame con los resultados de la varianza explicada y el radio de la varianza explicado.

    """
    df_pca_table1 = pd.DataFrame(data=modelo.explained_variance_, columns=['Explained Variance'])
    df_pca_table2 = pd.DataFrame(data=modelo.explained_variance_ratio_, columns=['Explained Variance Ratio'])
    componentes = ['PC ' + str(i + 1) for i in range(modelo.n_components)]
    df_componentes = pd.DataFrame(data=componentes, columns=['Principal_Component'])
    df_pca_table = pd.concat([df_componentes,df_pca_table1,df_pca_table2], axis=1)
    return(df_pca_table)


####################################################################################################################################

def Graficar_varianza_PCA(modelo):
    """
    Funcion para generar las graficas de la varianza explicada y el radio de la varianza explicada

    Parameters
    ----------
    modelo : sklearn.model
        Modelo ajustado (tipo de dato propio de sklearn).

    Returns
    -------
    Figura y ejes. Graficas de la varianza explicada, ratio de la varianza explicada y varianza acumulada

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.tight_layout(pad=5)
    ax1.set_title('Explained Variance (Absolute)')
    ax1.set(ylabel='Explained Variance', xlabel='Components')
    ax1.bar(range(1,len(modelo.explained_variance_ )+1),modelo.explained_variance_ )
    ax1.plot(range(1,len(modelo.explained_variance_ )+1), np.cumsum(modelo.explained_variance_), color='red', marker='o')
    ax2.set_title('Explained Variance (Relative)')
    ax2.set(ylabel='Explained Variance Ratio', xlabel='Components')
    ax2.plot(range(1,len(modelo.explained_variance_ )+1), modelo.explained_variance_ratio_, marker='o', color='green')
    return(fig, ax1, ax2)


####################################################################################################################################

def Remove_Outliers(Data, Label):
    """
    Funcion para remover los valores atipicos de un conjunto de datos. Se emplea el criterio de Tukey donde todos los valores 
    por encima de 1.5 veces el rango intercuantil mas el tercer cuantil se remueven asi como los valores 1.5 veces el rango 
    intercuantil menos el primer cuantil.

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos a los que se les removeran los atipicos.
    Label : String
        Nombre de la columna con los datos a los cuales se les removeran los atipicos.

    Returns
    -------
    pd.DataFrame con las etiquetas de valores atipicos.

    """
    Q1 = Data[Label].quantile(0.25); Q3 = Data[Label].quantile(0.75); IQR = Q3 - Q1
    Data['outlier_%s' %Data[Label].name] = [1 if x > Q3 + 1.5*IQR or x < Q1 - 1.5*IQR else 0 for x in Data[Label]]
    return(Data)


####################################################################################################################################

def Plot_Hist(Data, n_bins, units='', size=10):
    """
    Funcion para graficar el histograma y boxplot de los datos
    
    Parameters
    ----------
    Data : pd.Series
        Datos con los cuales se construira el histograma y boxplot.
    n_bins : int
        Para el histograma, numero de bins.
    units : str, optional
        Unidades de los datos. The default is ''.
    size : int, optional
        Tamaño de la grafica. The default is 10.

    Returns
    -------
    Figura. Grafica del histograma y boxplot para la pd.Series proporcionada

    """
    ## Para definir el numero de graficas (2 en este caso), el tamaño de la figura y la relacion de aspecto entre las graficas.
    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 5]}, figsize=(size, size*0.75), sharex=True)
    ## Titulo de la grafica
    fig.suptitle('Histogram & Boxplot of %s' % Data.name, size=size*2)
    ## Para graficar el Boxplot
    axs[0].boxplot(Data, vert=False, meanline=True, showmeans=True, widths=0.70, patch_artist=True, 
                   boxprops={'facecolor':'g'}, medianprops={'color':'blue','linestyle':'--'})
    axs[0].set_yticks([]);
    ## Para graficar el histograma.
    fig1 = axs[1].hist(Data, bins=n_bins, edgecolor='black', facecolor='grey')
    ## Para añadir las etiquetas a las barras (conteo absoluto)
    bin_centers = np.diff(fig1[1])*0.5 + fig1[1][:-1]
    n = 0
    for fr, x, patch in zip(fig1[0], bin_centers, fig1[2]):
        height = int(fig1[0][n])
        plt.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2),
                     textcoords = "offset points", ha = 'center', va = 'bottom', size=12)
        n = n+1
    ## Para añadir las etiquetas de los ejes.
    axs[1].set_xlabel(Data.name + '\n%s' %units)
    axs[1].set_ylabel('Absolute Frequency')
    ## Se añaden las lineas verticales de la media y la mediana a ambas graficas.
    axs[0].axvline(x=Data.mean(), color='r', linestyle=':', label='Mean')
    axs[0].axvline(x=Data.median(), color='darkblue', linestyle='-', label='Median')
    axs[1].axvline(x=Data.mean(), color='r', linestyle=':', label='Mean')
    axs[1].axvline(x=Data.median(), color='darkblue', linestyle='-', label='Median')
    ## Parametros miscelaneos (Posicion y tamaño de la leyenda, espacio entre las graficas)
    axs[1].legend(loc='upper right', fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0)
    return(fig,axs)


#####################################################################################################################################

def Boxplots(Data,Labels,Limits='auto',figsize=[8,10]):
    """
    Funcion para calcular boxplots de una serie de datos dada.

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos con las propiedades a las que se les graficara el boxplot.
    Labels : List
        Lista de propiedades para graficar los boxplots. Los boxplots se generan en el mismo orden que esta variable. 
        Los nombres deben coincidir exactamente con los nombres de las columnas en el pd.DataFrame
    Limits : list, optional
        Valores para los limites del eje y en los boxplots. The default is 'auto' y este valor usa el minimo y el maximo de la
        primer propiedad graficada.
    figsize : list, optional
        Ancho y alto de la grafica. The default is [8,10].

    Returns
    -------
    Figuras y ejes.

    """
    Data.dropna(axis=0, inplace=True)
    fig, axs = plt.subplots(1, len(Labels), figsize=figsize, sharey=True)
    title = 'Boxplots of {log}'.format(log=[Label for Label in Labels])
    fig.suptitle(title, size=20, y=0.92)
    
    for num, label in enumerate(Labels):
        if len(Labels) == 1:
            axs.boxplot(Data[label],
                            vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True, 
                            boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':2},
                            meanprops={'color':'red','linewidth':2})
            axs.set_title('%s' %label, y=-0.04)
            if Limits == 'auto':
                axs.set_ylim([Data[label].min(),Data[label].max()])
            else:
                axs.set_ylim(Limits[0], Limits[1])
                
            axs.set_xticks([]); axs.set_ylabel(label)
        else:
            axs[num].boxplot(Data[label],
                             vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True,
                             boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':2},
                            meanprops={'color':'red','linewidth':2})
            axs[num].set_title('%s' %label, y=-0.04)
            axs[num].set_xticks([])
        
        plt.subplots_adjust(wspace=0, hspace=0)
    
    return(fig,axs)


#####################################################################################################################################

def Class_Boxplots(Data,Label,Clase,Limits='auto',figsize=[8,10],median=True,mean=True):
    """
    Funcion para graficar boxplots de una propiedad filtrada por una variable categorica (clase)

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos, deben incluir la variable para contruir los boxplots y la clase para filtrar.
    Label : string
        Nombre de la propiedad para graficar, debe coincidir con el nombre de la columna en el pd.DataFrame.
    Clase : string
        Nombre de la variable categorica para filtrar los boxplots, debe coincidir con el nombre de la columna en el pd.DataFrame.
    Limits : list, optional
        Valores para los limites del eje y. The default is 'auto' y utiliza el valor maximo y minimo de la propiedad.
    figsize : list, optional
        Valores de ancho y altura para generar la grafica. The default is [8,10].
    median : bool, optional
        Opcion para marcar el valor de la mediana global. The default is True.
    mean : bool, optional
        Opcion para marcar el valor de la media global. The default is True.

    Returns
    -------
    Figura y ejes.

    """

    Data_no_nan = Data.copy()
    #Data_no_nan.dropna(axis=0, inplace=True)
    fig, axs = plt.subplots(1,Data_no_nan[Clase].value_counts().shape[0], figsize=figsize, sharey=True)
    title = 'Class filtered Boxplot of {log} ({class_name})'.format(log=Label,class_name=Clase)
    fig.suptitle(title, size=20, y=0.92)
    for num, class_label in enumerate(Data_no_nan[Clase].value_counts().sort_index(ascending=True).index):
        if num == 0:
            axs[num].boxplot(Data_no_nan[Data_no_nan[Clase] == class_label][Label],
                            vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True, 
                            boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':1},
                            meanprops={'color':'red','linewidth':1})
            axs[num].set_title('Class_%i' %class_label, y=-0.04)
            if Limits == 'auto':
                axs[num].set_ylim([Data_no_nan[Label].min(),Data_no_nan[Label].max()])
            else:
                axs[num].set_ylim(Limits[0],Limits[1])
            axs[num].set_xticks([]); axs[num].set_ylabel(Label)
            if mean == True:
                axs[num].axhline(y=Data_no_nan[Label].mean(), color='r', linestyle=':', label='Mean')
                axs[num].legend(loc='upper left', fontsize='x-small')
            if median == True:
                axs[num].axhline(y=Data_no_nan[Label].median(), color='darkblue', linestyle='-', label='Median')
                axs[num].legend(loc='upper left', fontsize='x-small')
        else:
            axs[num].boxplot(Data_no_nan[Data_no_nan[Clase] == class_label][Label],
                             vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True,
                             boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':1},
                            meanprops={'color':'red','linewidth':1})
            axs[num].set_title('Class_%i' %class_label, y=-0.04)
            axs[num].set_xticks([])
            if mean == True:
                axs[num].axhline(y=Data_no_nan[Label].mean(), color='r', linestyle=':', label='Mean')
            if median == True:
                axs[num].axhline(y=Data_no_nan[Label].median(), color='darkblue', linestyle='-', label='Median')

    plt.subplots_adjust(wspace=0, hspace=0)
    return(fig, axs)


#####################################################################################################################################

def Silhouette_plots(Data, No_clusters):
    """
    Funcion para graficar los resultados de la implementacion del metodo de la silueta. 
    Se utiliza el metodo de KMeans para estimar los clusters y sus coeficientes de silueta.

    Parameters
    ----------
    Data : pd.DataFrame
        Datos de las caracteristicas (features) a las que se les calculara el coeficiente de la silueta.
    No_clusters : list of int
        Numero de clusters para los que se estimara el coeficiente de la silueta.

    Returns
    -------
    Figura. Graficas de dispersion y del coeficiente de silueta para cada numero de clusters.

    """
    ## Se inicializa la grafica en funcion de los numeros de clusters.
    range_n_clusters = No_clusters
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(Data.to_numpy()) + (n_clusters + 1) * 10])
    ## Se generan los clusters mediante el metodo de KMeans.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(Data.to_numpy())
    ## Se obtiene el coeficiente de la silueta, se imprimen los valores en pantalla y en un pd.DataFrame.
        silhouette_avg = silhouette_score(Data.to_numpy(), cluster_labels)
        print("For n_clusters =", n_clusters, 
              "The average silhouette_score is :", silhouette_avg,)
        sample_silhouette_values = silhouette_samples(Data.to_numpy(), cluster_labels)
    ## Se crea el grafico de dispersion (Scatterplot) para visualizar los datos y los clusters. El grafico de dispersion
    ## se construye con las dos primeras caracteristicas (features) introducidas en los Datos
        y_lower = 10
        for i in range(n_clusters):
        
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7,)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ## Se marca con una linea recta el maximo valor del coeficiente de silueta para cada grafica.
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ## Se etiquetan y dibujan los centros de los clusters en la grafica
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(Data.to_numpy()[:, 0], Data.to_numpy()[:, 1], 
                    marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k",)
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        ## Parametros miscelaneos 
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters,
                     fontsize=14, fontweight="bold",)
    return(fig, (ax1, ax2))


####################################################################################################################################

def Silhouette(Data,ks=range(2, 30)):
    """
    Funcion para generar una grafica de coeficientes de silueta, utilizada durante el proceso de analisis
    del numero optimo de clusters

    Parameters
    ----------
    Data : pd.DataFrame
        Datos a partir de los cuales se generara la grafica.
    ks : list, optional
        Lista de numeros a los cuales se les estimara el coeficiente de silueta. The default is range(2, 30).

    Returns
    -------
    Grafica y tiempo de ejecucion.

    """
    start_time = time.time()
    silhouette=[]

    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = silhouette_score(Data, labels, metric='euclidean', sample_size=50000)
        silhouette.append(score)
    elapsed_time = time.time() - start_time
    df_silhouette = pd.DataFrame(data=zip(ks,silhouette), columns=['N_Clusters','Silhouette_score'])
    min_score1 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[0]
    min_score2 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[1]
    min_score3 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[2]

    print(f"Elapsed time to perform silhouette analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")

    plt.title('Método de silueta')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, silhouette, '*-', label='Silhouette')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Coeficiente de silueta')
    plt.legend()
    return()


####################################################################################################################################

def CH_Index(Data,ks=range(2,30)):
    """
    Funcion para obtener y graficar el indice de Calinsky-Harabasz, empleada para el analisis del numero optimo de clusters.

    Parameters
    ----------
    Data : pd.DataFrame
        Datos con los cuales se realizara el proceso de agrupamiento (clustering).
    ks : list, optional
        Numeros de clusters para estimar el coeficiente de Calinsky-Harabasz. The default is range(2,30).

    Returns
    -------
    Grafica del coeficiente de Calinsky-Harabaz contra el numero de clusters.

    """
    start_time = time.time()
    vrc=[]
    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = calinski_harabasz_score(Data, labels)
        vrc.append(score)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to perform vrc analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")
    df_CH = pd.DataFrame(data=zip(ks,vrc), columns=['N_Clusters','Calinsky_Harabaz_score'])
    min_score1 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[0]
    min_score2 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[1]
    min_score3 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[2]

    plt.title('Índice de Calinski-Harabasz (Variance Ratio Criterion, VRC)')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, vrc, '*-', label='VRC')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Índice de Calinski-Harabasz')
    plt.legend()
    plt.show()
    return()


####################################################################################################################################

def DB_Index(Data,ks=range(2, 30)):
    """
    Funcion para graficar el indice de Davies-Bouldin, empleado en el proceso de determinacion del numero optimo de clusters

    Parameters
    ----------
    Data : pd.DataFrame
        Datos a partir de los cuales se realizara el agrupamiento.
    ks : list, optional
        Numeros de clusters para los que se estimara el coeficiente de Davies-Bouldin. The default is range(2, 30).

    Returns
    -------
    Grafica con los coeficientes de Davies-Bouldin contra el numero de clusters.

    """
    start_time = time.time()
    dbi=[]

    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = davies_bouldin_score(Data, labels)
        dbi.append(score)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to perform dbi analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")
    df_DB = pd.DataFrame(data=zip(ks,dbi), columns=['N_Clusters','Davies_Bouldin_score'])
    min_score1 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[0]
    min_score2 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[1]
    min_score3 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[2]
    
    plt.title('Índice de Davies-Bouldin')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, dbi, '*-', label='Davies-Bouldin')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Índice de Davies-Bouldin')
    plt.legend()
    plt.show()
    return()


#################################################################################################################################### 

def optimalK(data, nrefs=3, maxClusters=15):
    """
    Funcion para estimar la grafica de compactamiento por el metodo de "Gap Statistics". 
    El metodo de clustering de referencia es KMeans.

    Parameters
    ----------
    data : pd.DataFrame
        Datos de las caracteristicas (features) a las que se les calculara el coeficiente de la silueta.
    nrefs : int, optional
        Numero de sub conjuntos de datos de referencia para la estimacion del "Gap Statistics". The default is 3.
    maxClusters : int, optional
        Numero maximo de clusters para la implementacion del metodo. The default is 15.

    Returns
    -------
    pd.DataFrame con los pares numero de cluster - gap statistics.

    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (resultsdf)


#####################################################################################################################################

def Graficar_Set(Datos,Entrenamiento,Validacion, Etiqueta):
    """
    Funcion para realizar el conteo de clases para los sets de entrenamiento y validacion durante el proceso de implementacion de
    metodos supervisados junto con el conteo de clases del conjunto de datos completo.

    Parameters
    ----------
    Datos : pd.DataFrame
        Conjunto de datos original.
    Entrenamiento : pd.DataFrame
        Set de datos de entrenamiento.
    Validacion : pd.DataFrame
        Set de datos de validacion.
    Etiqueta : string
        Nombre de la columna con la clasificacion.

    Returns
    -------
    Figura y ejes con los histogramas y el conteo de clases.

    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.tight_layout()
    axes[0] = sns.barplot(ax=axes[0], data=Datos, 
                x=Datos[Etiqueta],
                y=Datos['COUNT']).set(title='Original DataFrame')
    axes[1] = sns.barplot(ax=axes[1], data=Entrenamiento, 
                x=Entrenamiento[Etiqueta], 
                y=Entrenamiento['COUNT']).set(title='Training DataFrame')
    axes[2] = sns.barplot(ax=axes[2], data=Validacion, 
                x=Validacion[Etiqueta], 
                y=Validacion['COUNT']).set(title='Testing DataFrame')
    return(fig,axes)


#####################################################################################################################################

def Plot_Well(Data,title,logs,colors,units,ref='DEPTH',ref_units='m',size=[17,17],median=False,mean=False):
    """
    Funcion para graficar registros geofisicos de pozo

    Parameters
    ----------
    Data : pd.DataFrame
        Datos de registros geofisicos para graficar. El pd.DataFrame debe contener una columna de profundidad 
        (no necesariamente la primer columna) y al menos un registro para graficar.
    title : string
        Titulo que tendra la grafica.
    logs : list
        Registros que seran graficados. El orden de los registros en la lista es el mismo que el de la figura de salida.
    colors : list
        Colores que se usaran para cada una de las curvas. Se deben colocar un color para cada registro.
    units : list
        Unidades de cada registro. Se debe colocar unidades para cada registro.
    ref : string, optional
        Nombre de la columna con los datos de profundidad. The default is 'DEPTH'.
    ref_units : string, optional
        Unidades en las que se encuentra la profundidad. The default is 'm'.
    size : list, optional
        Tamaño de la grafica, se deben colocar dos valores correspondientes a alto y ancho. The default is [17,17].
    median : bool, optional
        Activar en caso de querer graficar el registro junto a la mediana. The default is False.
    mean : bool, optional
        Activar en caso de querer graficar el registro junto a la media. The default is False.

    Returns
    -------
    Figura y ejes.

    """
    
    ## Se deine el tamaño de los carriles, titulo de la grafica y se elimina el espacio entre carriles
    fig, axes = plt.subplots(1,len(logs),figsize=size,sharey=True)
    fig.suptitle(title ,fontsize=24, x = 0.5, y = 0.95,ha = 'center')
    fig.subplots_adjust(wspace = 0.0)
    
    ## Para cada una de las graficas, se configuran las distintas partes de la grafica
    for num,ax in enumerate(axes):
        if num == 0:
            ## Se definen las mallas de la curva que se va a graficar.
            axes[num] = plt.subplot2grid((1,len(logs)), (0,num), rowspan=1, colspan = 1)
            ## Se definen el registro que se va a graficar.
            axes[num].plot(Data[logs[num]], Data[ref], c=colors[num], lw=0.5)
            ## Se establecen los limites de la grafica, tanto el registro como la profundidad.
            axes[num].set_xlim(math.floor(min(Data[logs[num]])), math.ceil(max(Data[logs[num]])))
            axes[num].set_ylim(math.floor(max(Data[ref])), math.floor(min(Data[ref])))
            ## Se establece la posicion de las escalas.
            axes[num].xaxis.set_ticks_position("top"); axes[num].xaxis.set_label_position("top")
            ## Se define el titulo de cada uno de los carriles.
            axes[num].set_ylabel(ref + ' (%s)' %ref_units)
            axes[num].set_xlabel(logs[num] + '\n' + units[num]); axes[num].grid()
            ## Se añade (opcional) el valor de la mediana
            if median == True:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='-', label='Median')
                axes[num].legend(loc='best', fontsize='medium')
            if mean == True:
                axes[num].axvline(x=Data[logs[num]].mean(),color='r', linestyle=':', label='Mean')
                axes[num].legend(loc='best', fontsize='medium')
        else:
            ## Se definen las mallas de la curva que se va a graficar.
            axes[num] = plt.subplot2grid((1,len(logs)), (0,num), rowspan=1, colspan = 1, sharey=axes[0])
            ## Se definen el registro que se va a graficar.
            axes[num].plot(Data[logs[num]], Data[ref], c=colors[num], lw=0.5)
            ## Se establecen los limites de la grafica, tanto el registro como la profundidad.
            axes[num].set_xlim(math.floor(min(Data[logs[num]])), math.ceil(max(Data[logs[num]])))
            axes[num].set_ylim(math.floor(max(Data[ref])), math.floor(min(Data[ref])))
            ## Se establece la posicion de las escalas.
            axes[num].xaxis.set_ticks_position("top"); axes[num].xaxis.set_label_position("top")
            ## Se define el titulo de cada uno de los carriles.
            axes[num].set_xlabel(logs[num] + '\n' + units[num]); axes[num].grid()
            if median == True:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='-', label='Median')
                axes[num].legend(loc='best', fontsize='medium')
            if mean == True:
                axes[num].axvline(x=Data[logs[num]].mean(),color='r', linestyle=':', label='Mean')
                axes[num].legend(loc='best', fontsize='medium')
                
    ## Se definen los ejes que no se mostraran
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible = False)  
    
    return(fig,axes)


#####################################################################################################################################

def Plot_Well_Fill(Data,Clase,Paleta_colores,title,logs,colors,units,ref='DEPTH',ref_units='m',size=[17,17],median=False,mean=False):
    """
    Función para graficar registros geofisicos de pozo con relleno en función de alguna variable categorica

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos donde se require al menos una variable continua (registro geofisico de pozo) y una variable categorica.
    Clase : str
        Nombre de la variable categorica que se usara para rellenar las curvas del registro.
    Paleta_colores : dict
        Diccionario con los pares.
    title : TYPE
        DESCRIPTION.
    logs : TYPE
        DESCRIPTION.
    colors : TYPE
        DESCRIPTION.
    units : TYPE
        DESCRIPTION.
    ref : TYPE, optional
        DESCRIPTION. The default is 'DEPTH'.
    ref_units : TYPE, optional
        DESCRIPTION. The default is 'm'.
    size : TYPE, optional
        DESCRIPTION. The default is [17,17].
    median : TYPE, optional
        DESCRIPTION. The default is False.
    mean : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    patches = [ mpatches.Patch(color=list(Paleta_colores.keys())[i],
                           label="Grupo {l}".format(l=list(Paleta_colores.values())[i]) ) for i in range(len(Paleta_colores)) ]
    
    # Se deine el tamaño de los carriles
    fig, axes = plt.subplots(1,len(logs), figsize=size, sharey=True)
    fig.suptitle(title ,fontsize=24, x = 0.5, y = 0.95,ha = 'center')
    fig.subplots_adjust(wspace = 0.0)
    
    ## Para cada uno de los registros, se configuran las distintas partes de la grafica
    for num,ax in enumerate(axes):
        if num == 0:
            ## Se definen las mallas de la curva que se va a graficar.
            axes[num] = plt.subplot2grid((1,len(logs)), (0,num), rowspan=1, colspan = 1)
            ## Se definen el registro que se va a graficar.
            axes[num].plot(Data[logs[num]], Data[ref], c=colors[num], lw=0.5)
            ## Se añade el relleno a la grafica. Se usa un ciclo for para pintar cada una de las facies con su respectivo color (ECGR)
            for color in Paleta_colores:
                axes[num].fill_betweenx(Data[ref], math.ceil(max(Data[logs[num]])), 
                      Data[logs[num]], where=Data[Clase] == Paleta_colores[color], color=color,
                      interpolate=True)
            ## Se establecen los limites de la grafica, tanto el registro como la profundidad.
            axes[num].set_xlim(math.floor(min(Data[logs[num]])), math.ceil(max(Data[logs[num]])))
            axes[num].set_ylim(math.floor(max(Data[ref])), math.floor(min(Data[ref])))
            ## Se establece la posicion de las escalas.
            axes[num].xaxis.set_ticks_position("top"); axes[num].xaxis.set_label_position("top")
            ## Se define el titulo de cada uno de los carriles.
            axes[num].set_ylabel(ref + ' (%s)' %ref_units)
            axes[num].set_xlabel(logs[num] + '\n' + units[num]); axes[num].grid()
            ## Se añade (opcional) el valor de la mediana
            if median == True:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='-', label='Median')
                axes[num].legend(loc='best', fontsize='medium')
            if mean == True:
                axes[num].axvline(x=Data[logs[num]].mean(),color='r', linestyle=':', label='Mean')
                axes[num].legend(loc='best', fontsize='medium')
        elif num % 2 == 0:
            ## Se definen las mallas de la curva que se va a graficar.
            axes[num] = plt.subplot2grid((1,len(logs)), (0,num), rowspan=1, colspan = 1, sharey=axes[0])
            ## Se definen el registro que se va a graficar.
            axes[num].plot(Data[logs[num]], Data[ref], c=colors[num], lw=0.5)
            ## Se añade el relleno a la grafica. Se usa un ciclo for para pintar cada una de las facies con su respectivo color (ECGR)
            for color in Paleta_colores:
                axes[num].fill_betweenx(Data[ref], math.ceil(max(Data[logs[num]])), 
                      Data[logs[num]], where=Data[Clase] == Paleta_colores[color], color=color,
                      interpolate=True)
            ## Se establecen los limites de la grafica, tanto el registro como la profundidad.
            axes[num].set_xlim(math.floor(min(Data[logs[num]])), math.ceil(max(Data[logs[num]])))
            axes[num].set_ylim(math.floor(max(Data[ref])), math.floor(min(Data[ref])))
            ## Se establece la posicion de las escalas.
            axes[num].xaxis.set_ticks_position("top"); axes[num].xaxis.set_label_position("top")
            ## Se define el titulo de cada uno de los carriles.
            axes[num].set_xlabel(logs[num] + '\n' + units[num]); axes[num].grid()
            if median == True:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='-', label='Median')
                axes[num].legend(loc='best', fontsize='medium')
            if mean == True:
                axes[num].axvline(x=Data[logs[num]].mean(),color='r', linestyle=':', label='Mean')
                axes[num].legend(loc='best', fontsize='medium')
        else:
            ## Se definen las mallas de la curva que se va a graficar.
            axes[num] = plt.subplot2grid((1,len(logs)), (0,num), rowspan=1, colspan = 1, sharey=axes[0])
            ## Se definen el registro que se va a graficar.
            axes[num].plot(Data[logs[num]], Data[ref], c=colors[num], lw=0.5)
            ## Se añade el relleno a la grafica. Se usa un ciclo for para pintar cada una de las facies con su respectivo color (ECGR)
            for color in Paleta_colores:
                axes[num].fill_betweenx(Data[ref], math.ceil(max(Data[logs[num]])), 
                      Data[logs[num]], where=Data[Clase] == Paleta_colores[color], color=color,
                      interpolate=True)
            ## Se establecen los limites de la grafica, tanto el registro como la profundidad.
            axes[num].set_xlim(math.ceil(max(Data[logs[num]])), math.floor(min(Data[logs[num]])))
            axes[num].set_ylim(math.floor(max(Data[ref])), math.floor(min(Data[ref])))
            ## Se establece la posicion de las escalas.
            axes[num].xaxis.set_ticks_position("top"); axes[num].xaxis.set_label_position("top")
            ## Se define el titulo de cada uno de los carriles.
            axes[num].set_xlabel(logs[num] + '\n' + units[num]); axes[num].grid()
            if median == True:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='-', label='Mediana')
                axes[num].legend(loc='best', fontsize='medium')
            if mean == True:
                axes[num].axvline(x=Data[logs[num]].mean(),color='r', linestyle=':', label='Media')
                axes[num].legend(loc='best', fontsize='medium')
                
    ## Se definen los ejes que no se mostraran
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible = False) 
    ## Se configura la leyenda de la grafica
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. ,fontsize=15)
    
    return(fig,axes)


#####################################################################################################################################













