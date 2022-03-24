import pandas as pd
from scipy.stats import chi2_contingency
import random 

def test_chiquadro(dataset,variabile_target, lista_variabili, index ):
    
    """ Data una variabile target e una lista di variabili indipendenti, si ottiene 
    il test del chi2 tra la variabile target e tutte le variabili di interesse e i 
    relativi parametri di interesse in un dataframe
    
    Richiede la libreria scipy e la libreria pandas 
    
    --------------------------
    Parametri: 
    dataset: il dataset di riferimento
    variabile_target: la variabile risposta 
    lista_variabili: lista delle variabili indipendenti di interesse (lista anche se fosse una)
    index: lista degli indici da attribuire a ciascuna osservazione (cioè le diverse variabili)
    
    --------------------------
    Output: dataframe con valore del chi2, p_value, gradi di libertà, e la significativa al 5%
    del test """
    
    chi2, p_value, dof, ex = [],[],[],[]
    
    for variabile in lista_variabili:
        chi2.append(chi2_contingency(pd.crosstab(variabile_target,variabile))[0])
        p_value.append(chi2_contingency(pd.crosstab(variabile_target,variabile))[1])
        dof.append(chi2_contingency(pd.crosstab(variabile_target,variabile))[2])
        ex.append(chi2_contingency(pd.crosstab(variabile_target,variabile))[3])
    chi2_dataset = pd.DataFrame(data = {"chi2": chi2,
                                        "p_value":p_value,
                                        "degree of freedom":dof}, 
                                index = index
                                         )
    sign_0 = [] 
    for i in range(len(chi2_dataset)):
        if chi2_dataset["p_value"][i] < 0.05:
            sign_0.append(True)
        else:
            sign_0.append(False)
    chi2_dataset["sign_0.05"] = sign_0
    
    return chi2_dataset


def imputazione_proporzionale(dataset, variabile):
    
    """ Imputazione dei dati mancanti a ciascuna categoria
    secondo la proporzionalità dei casi all'interno del dataset 
    
    -------------------
    Parametri:
    dataset: il dataset di riferimento
    variabile: stringa della variabile per cui vogliamo imputare dati mancanti 
    --------------------
    Output:
    dataset con i dati imputati alla variabile di riferimento """
    
    e = dataset[variabile].dropna().value_counts()/len(dataset[variabile].dropna())*100
    len_index =  len(dataset[dataset[variabile].isna()].index)
    for i in range(len(e)):
        print("""Il {:.2f}% delle osservazioni appartiene alla categoria "{}" """.format(e[i],e.index[i]))
    random.seed(42)
    dataset_index = dataset[dataset[variabile].isna()].index
    
    for categoria,valore in zip(e.index,e):
        idx_Pr = random.sample(list(dataset_index), k = round(valore*len_index/100))
        dataset.loc[idx_Pr] = dataset.fillna({variabile : categoria})
        dataset_index = dataset_index.drop(idx_Pr) 
 
    return dataset
    
    
    