import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import kendalltau,chi2_contingency, pearsonr
import pandas as pd

def plot_var_num(dataset,variabile):
    """ Plot delle variabili numeriche 
    --------------
    Parametri: 
    dataset: dataset di riferimento
    variabile: variabile da plottare
    --------------
    Output
    ritorna l'istogramma ed il boxplot della variabile
    Se assegnata ad un elemento python ritorna i quartili, valore medio, massimo e medio
    della distribuzione"""
    
    descrizione = dataset[variabile].describe()
    plt.figure(figsize = (14,8))
    plt.subplot(1,2,1)
    sns.histplot(dataset[variabile], color = "red")
    plt.title("Istogramma della variabile {}".format(variabile))
    plt.subplot(1,2,2)
    sns.boxplot(dataset[variabile])
    plt.title("Boxplot della variabile {}".format(variabile))
    plt.show()
    
    return descrizione

def dipendenza_correlazione(dataset, variabile1, variabile2, p_value_lv = 0.05):
    """ Funzione che restituisce la dipendenza tra due variabili
    ---------------------
    Parametri:
    dataset:dataset di riferimento
    variabile1, variabile2: stringa del nome delle variabili su cui attuare il test 
    p_value_lv: livello di significatività
    ---------------------
    Output: 
    Date due variabili numeriche
    dataframe contentente:
    * il coefficiente di correlazione di Perason
    * p_value associato
    * Booleano che indica se il test è significativo per un livello pari a p_value_lv
    
    Date due variabili categoriche
    dataframe contentente:
    * testchi2
    * p_value associato
    * Booleano che indica se il test è significativo per un livello pari a p_value_lv
    
    Date due variabili mischiate
    dataframe contenente
    * coefficiente di correlazione di kendall
    * p_value associato
    * Booleano che indica se il test è significativo per un livello pari a p_value_lv
    """
    
    if dataset[variabile1].dtypes == "int64" and  dataset[variabile2].dtypes == "int64":
        p_value = pearsonr(dataset[variabile1], dataset[variabile2])[1]
        corr = pearsonr(dataset[variabile1], dataset[variabile2])[0]
        data = pd.DataFrame(data = {"corr": [corr], "p_value": [p_value]}, index = ["Pearson"])
        sign_0 = [] 
        if data["p_value"][0] < p_value_lv:
            sign_0.append(True)
        else:
            sign_0.append(False)
        data["sign_{}".format(p_value_lv)] = sign_0


    elif dataset[variabile1].dtypes == "object" and  dataset[variabile2].dtypes == "object":
        p_value = chi2_contingency(pd.crosstab(dataset[variabile1], dataset[variabile2]))[1]
        data = pd.DataFrame(data = {"p_value": [p_value]}, index = ["chi2"])
        
        sign_0 = [] 
        if data["p_value"][0] < p_value_lv:
            sign_0.append(True)
        else:
            sign_0.append(False)
        data["sign_{}".format(p_value_lv)] = sign_0
    else:
        correlation = kendalltau(dataset[variabile1],dataset[variabile2])[0]
        p_value = kendalltau(dataset[variabile1],dataset[variabile2])[1]
        
        data = pd.DataFrame(data = {"correlation": [correlation],"p_value":[p_value]},
                           index = ["Kendall"])
        sign_0 = [] 
        if data["p_value"][0] < p_value_lv:
            sign_0.append(True)
        else:
            sign_0.append(False)
        data["sign_{}".format(p_value_lv)] = sign_0
        
    return data
        
def analisi_variabili_categoriche(dataset, variabile1, variabile2, normalize = "index"):
    
    """ Resituisce due grafici: 
        * il barplot della variabile1 
        * il barplot della variabile1 condizionata alla variabile2
        * Se assegnata ad un elemento Python ritorna la quantità di 
        osservazioni per ciascuna categoria della variabile1
        * Se assegnata ad un elemento Python ritorna la tabella di 
        contingenza tra la variabile1 e la variabile 2 
     ----------------------------
     Parametri:
     dataset: dataset di riferimento
     variabile1: stringa del nome della variabile per cui disegnare i grafici
     variabile2: stringa del nome della variabile di condizionamento 
     normalize: stringa che permette di decidere come indicizzare la colonna
     ["index" per riga, "column" per colonna, "all" per entrambi, "False" nessuna
     normalizzazione]
     
     """
    conteggio = dataset[variabile1].value_counts()/len(dataset)
    tabella = pd.crosstab(dataset[variabile1], dataset[variabile2], normalize = normalize)
    plt.figure(figsize =(15,22))
    plt.subplot(2,1,1)
    sns.countplot(y = dataset[variabile1])
    plt.title("""Barplot della variabile "{}" """.format(variabile1))
    plt.subplot(2,1,2)
    sns.countplot(y = dataset[variabile1], hue = dataset[variabile2])
    plt.title("""Barplot della variabile "{}" condizionata alla variabile {} """.format(variabile1,variabile2))
        
    return conteggio,tabella 


def unificazione_categorie(dataset, variabile, categoria_da_trasf, trasformazione):
    """ Unifica due o più categorie con la stessa stringa
    ------------------
    Parameters:
    dataset: dataset di riferimento
    variabile: stringa della variabile per la quale avverrà il cambiamento delle categorie
    categoria_da_trasf: lista della/e categorie sulla quale applicare il cambiamento
    trasformazione: stringa che indica la categoria da attribuire
    
    ------------------
    Output:
    dataset trasformato """
    for categoria in categoria_da_trasf:
        dataset = dataset.replace(categoria, trasformazione)
    return dataset

def histplot_per_categorie(dataset, variabile1, variabile_divisione):
    """ Plot che ritorna più istogrammi della variabile a cui siamo interessati 
    che suddividono il dataset nelle diverse categorie della variabile scelta
    per la divisione 
    
    -----------------
    Parametri: 
    dataset: dataset di riferimento
    variabile1: stringa della variabile numerica per la quale siamo interessati 
    a conoscere la distribuzione
    variabile_divisione: stringa della variabile per la quale siamo interessati
    avvenga la suddivisione del dataset
    
    -------------------
    """
        
    lunghezza = len(dataset[variabile_divisione].value_counts().index)
    plt.figure(figsize = (20,10))
    for i in range(1,lunghezza+1):
        plt.subplot(1, lunghezza,i)
        data = dataset[dataset[variabile_divisione] == dataset[variabile_divisione].value_counts().index[i-1]]
        sns.histplot(data[variabile1])
        plt.title("Istogramma della variabile '{}', data la categoria {}".format(variabile1,
                                       dataset[variabile_divisione].value_counts().index[i-1]))
        
        
def histplot_1_per_categorie(dataset, variabile1, variabile_divisione, x = 1, y = 0):
    
    """ Plot che ritorna più istogrammi della variabile a cui siamo interessati 
    che suddividono il dataset nelle diverse categorie della variabile scelta
    per la divisione 
    
    -----------------
    Parametri: 
    dataset: dataset di riferimento
    variabile1: stringa della variabile numerica per la quale siamo interessati 
    a conoscere la distribuzione
    variabile_divisione: stringa della variabile per la quale siamo interessati
    avvenga la suddivisione del dataset
    x,y: definiscono la suddivisione dei plot per riga e colonna rispettivamente 
    (Nota: il prodotto tra x e y deve essere uguale o maggiore alle categorie della 
    variabile per la quale avviene lo split)
    -------------------"""
    
    if y == 0:
        lunghezza = len(dataset[variabile_divisione].value_counts().index)
        plt.figure(figsize = (20,10))
        for i in range(1,lunghezza+1):
            plt.subplot(1, lunghezza,i)
            data = dataset[dataset[variabile_divisione] ==\
                       dataset[variabile_divisione].value_counts().index[i-1]]
            sns.histplot(data[variabile1])
            plt.title("Istogramma della variabile '{}', data la categoria {}".format(variabile1,
                                       dataset[variabile_divisione].value_counts().index[i-1]))
    else:
        lunghezza = len(dataset[variabile_divisione].value_counts().index)
        plt.figure(figsize = (20,10))
        for i in range(1,lunghezza+1):
            plt.subplot(x, y,i)
            data = dataset[dataset[variabile_divisione] == \
                           dataset[variabile_divisione].value_counts().index[i-1]]
            sns.histplot(data[variabile1])
            plt.title("Istogramma della variabile '{}', data la categoria {}"\
                      .format(variabile1,dataset[variabile_divisione].value_counts().index[i-1]))
              
