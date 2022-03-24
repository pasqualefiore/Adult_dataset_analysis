import numpy as np
import pandas as pd 

def caricamento_dataset(file_path, columns_names =["Age","Tipo_lavoro","fnlwgt","Istruzione","Anni_istruzione","Stato_sociale",
               "Occupazione","Condizione_familiare","Etnia","Genere","Cap_guadagnato",
               "Cap_perso","Ore_settimanali","Paese_nativo","Reddito"]):
    
    """ Permette il caricamento di un dataset csv attribuendo dei nomi alle variabili
    
    -----------
    Parametri: 
    file_path: il percorso necessario perchè python trovi il file csv
    columns_names: lista dei nomi delle variabili 
    
    ----------- """
    dataset = pd.read_csv(file_path,header = None, names = columns_names, na_values = " ?")
    
    return dataset

class informazioni_dataset():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def tipo_variabili(self, dataset):
        
        """ Otteniamo delle informazioni in merito alla natura delle variabili 
        ------------------
        
        Parametri: 
        dataset: il dataset di riferimento 
        
        -------------------------"""
        
        numeriche = 0
        categoriche = 0
        for i in range(len(self.dataset.columns)):
            if self.dataset[dataset.columns[i]].dtypes == "int64":
                numeriche += 1
            else: 
                categoriche += 1
        print("Ci sono {} variabili numeriche".format(numeriche))
        print("Ci sono {} variabili categoriche".format(categoriche))
        return numeriche, categoriche 
    
    def dati_mancanti(self, dataset):
        """ Definisce le variabili per le quali ci sono dati mancanti 
        e le quantità 
        -------------------
        Parametri:
        dataset: dataset che abbiamo a che fare 
        -------------------
        Output:
        dizionario variabile,quantità di osservazioni per i quali non abbiamo 
        dati 
        ---------------------"""
        
        lista_dati = {}
        dati_mancanti = self.dataset.isna().sum() 
        for i in range(len(dati_mancanti)):
            if dati_mancanti[i] != 0:
                lista_dati[dataset.columns[i]] = dati_mancanti[i]
                print("""Ci sono {} dati mancanti per la variabile {}""".format(dati_mancanti[i],dataset.columns[i]))
        return lista_dati 
    