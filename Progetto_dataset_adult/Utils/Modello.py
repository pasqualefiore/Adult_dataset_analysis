import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


import matplotlib.pyplot as plt

def feature_importances_plot(model, labels, **kwargs):
    """
    Calcola l'importanza normalizzata delle variabili di un modello
    restituendo i dati e mostrando il plot.
    -----------
    
    Parametri:
    model : modello di scikit-learn
    labels : list | np.array
        lista delle labels delle variabili
    
    Returns
    -------
    AxesSubplot
    """
    feature_importances = model.feature_importances_
    feature_importances = 100 * (feature_importances / feature_importances.max())
    df = pd.DataFrame(data={"feature_importances": feature_importances}, index=labels)
    df.sort_values(by="feature_importances", inplace=True)
    return df.plot(kind="barh", figsize=(8, 10), title=f"Feature importances", legend=None, **kwargs)

def divisione_x_y(dataset, target):
    
    """Permette la divisione del dataset in X e Y 
    
    ---------------
    Parametri
    dataset: dataset di riferimento
    target: stringa della variabile target
    
    ----------------
    Output
    Suddivisione X e Y
    ---------------- """
    X = dataset.drop(target, axis = 1)
    Y = dataset[target] 
    
    
    return X,Y

def trasformazioni_variabili(X, column_one_hot,column_categorical,
                             categories_categorical,column_normalizer, drop = "first", verbose = 1):
    
    """ Funzione che ritorna le variabili trasformate così come richiesto 
    ---------------
    Parametri: 
    X: dataset delle variabili indipendenti 
    column_one_hot: lista delle variabili a cui sarà applicata il One Hot encoding
    drop: decidere se droppare o meno le variabili [None,"first", "if binary"]
    column_categorical: lista delle variabili a cui sarà applicata l'Ordinal Encoding
    categories_categorical: lista di liste delle categorie per ciascuna variabili 
    a cui sarà applicata l'Ordinal Encoding (necessaria la lista di liste anche in casp
    di una sola variabile)
    column_normalizer: lista delle variabili a cui sarà applicata la normalizzazione
    
    ---------------
    Output: 
    arr: dataset trasformato
    ct: trsformazione delle colonne"""
    
    ct = ColumnTransformer(transformers=[
    ("ohe", OneHotEncoder(drop= drop, sparse=False, dtype=int),column_one_hot), 
    ("minmax", MinMaxScaler(), column_normalizer),
    ("oe", OrdinalEncoder(categories = categories_categorical,dtype = int),column_categorical)],
    remainder = "passthrough")
    
    arr = ct.fit_transform(X)
    if verbose == 1:
        print("La categorizzazione avverrà in questo modo {}".format(ct))
        print("""La shape del nuovo dataset è la seguente: {} osservazioni e {}    variabili""".format(arr.shape[0],arr.shape[1]))
    return arr, ct

def train_test_val(X,Y, validation = True, stratificazione = True, test_size = 0.20, val_size = 0.20,
                   random_state = 42):
    """ Funzione che ritorna la suddivisione in train, test o in train test 
    e validation della X e della Y
    
    ----------------------------
    Parametri:
    X,Y: dataset o array delle variabili X e della variabile Y
    test_size, val_size: valore compreso tra 0 e 1 che regola la 
    dimensione del test e del validation test
    random_state: randomicità dell'algoritmo
    validation: Booleano per comprendere se si vuole un validation 
    test o meno
    stratificazione: booleano che regola se stratificare il campione 
    in base ai valori della Y 
    --------------------------
    Output:
    train e test set se validation = False
    train, test e val set se validation = True
    """
    
    if stratificazione == True:
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify = Y, 
                                                     test_size = test_size)
    else:
         X_train, X_test, Y_train, Y_test = train_test_split(X,Y, 
                                                     test_size = test_size)
        
    if validation == True:
        if stratificazione == True:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, stratify = Y_train, 
                                                     test_size = round(len(X)*val_size))
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, 
                                                     test_size = round(len(X)*val_size))
            
        return X_train, X_test, X_val,Y_train, Y_test, Y_val
    else:
        return X_train, X_test,Y_train, Y_test
        
        
def label_encoder_val(Y_train, Y_test, Y_val ):
    """ Funzione che ritorna la variabile target trasformata
    -------
    Parametri: 
    Y_train: Train set della y
    Y_test: Test set della y 
    Y_val: Validation set della y
    ------
    Output:
    Y_train, Y_test, Y_val trasformati
    """
    
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.fit_transform(Y_test)
    Y_val = le.fit_transform(Y_val)
    
    return Y_train, Y_test,Y_val,le

def label_encoder_test(Y_train, Y_test):
    """ Funzione che ritorna la variabile target trasformata
    -------
    Parametri: 
    Y_train: Train set della y
    Y_test: Test set della y 
    ------
    Output:
    Y_train, Y_test,  trasformati
    """
    
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.fit_transform(Y_test)
    
    return Y_train, Y_test,le

def albero(X_train, Y_train, X_test, Y_test,X_val = [], Y_val = [], random = 42,
          feature_names =  ["Non occupied","Private","Self-empinc","Self-emp-notinc","Governative","Married",
                 "Spouse absent","No_relation","Widowed","Black","Other","White",
                 "Male","United_states","Istruzione","Age_log"],
          verbose = 1, plot = True, class_names = [], confusion_mat = True,
          criterion ='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
           min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,
           min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
    
    """Funzione che ritorna l'albero di classificazione e la matrice di confusione 
    
    ------------------
    Parametri:
    *: tutti i parametri dell'albero previsti da sklearn
    X_train, X_test, X_val: train, test e validation set della matrice delle X 
    Y_train, Y_test, Y_val: train, test e validation set della matrice delle Y
    verbose: Se valorizzato ad 1 fornisce informazioni degli score dei vari set
    feature_names: lista dei nomi delle categorie delle diverse features
    plot: Se True fornisce un plot dell'albero ottenuto
    confusion_mat: Se True ritorna un elemento a cui è possibili assegnare la matrice di confusione
    class_names: lista delle categorie della variabile y
    
    -------------------
    Output
    fit dell'albero
    matrice di confusione se confusion_mat = True
    ---------------------- """
    
    dtc = DecisionTreeClassifier(criterion = criterion, splitter = splitter, 
          max_depth = max_depth, min_samples_split = min_samples_split,
          min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf,
          max_features = max_features, random_state = random,
          max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, 
          class_weight = class_weight, ccp_alpha = ccp_alpha)
    dtc.fit(X_train, Y_train)
    feature_importances_plot(model=dtc, labels=feature_names)
    if verbose == 1:
        print("La profondità dell' albero è {}".format(dtc.get_depth()))
        if X_val == []:
            print("L'accuracy sul test set è pari a {}".format(dtc.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, dtc.predict(X_test))))
        else:
            print("L'accuracy sul validation test è pari a {}".format(dtc.score(X_val, Y_val)))
            print("L'accuracy sul test set è pari a {}".format(dtc.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, dtc.predict(X_test))))
                                             
    if plot == True:
        plt.figure(figsize=(20, 20))
        plot_tree(decision_tree=dtc, feature_names=feature_names, 
        class_names = class_names, filled=True, fontsize=12)
        plt.show()
              
    if confusion_mat == True: 
        confusion = confusion_matrix(Y_test, dtc.predict(X_test))
        print("Di seguito la matrice di confusione:\n{}".format(confusion))
        return dtc, confusion
    else:
        return dtc
                
        
def random_forest(X_train, Y_train, X_test, Y_test, X_val = [], Y_val = [],random = 42,
          feature_names =  ["Non occupied","Private","Self-empinc","Self-emp-notinc","Governative","Married",
                 "Spouse absent","No_relation","Widowed","Black","Other","White",
                 "Male","United_states","Istruzione","Age_log"],
          verbose = 1,class_names = [], confusion_mat = True, n_estimators=100,
          criterion ='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
           min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0,bootstrap=True, class_weight=None,oob_score=False, 
           warm_start=False, ccp_alpha=0.0,max_samples=None):
    
    
    """Funzione che ritorna il random forest di classificazione e e la matrice di confusione 
    
    ------------------
    Parametri:
    *: tutti i parametri dell'albero previsti da sklearn
    X_train, X_test, X_val: train, test e validation set della matrice delle X 
    Y_train, Y_test, Y_val: train, test e validation set della matrice delle Y
    verbose: Se valorizzato ad 1 fornisce informazioni degli score dei vari set
    feature_names: lista dei nomi delle categorie delle diverse features
    confusion_mat: Se True ritorna un elemento a cui è possibili assegnare la matrice di confusione
    class_names: lista delle categorie della variabile y
    
    -------------------
    Output
    fit dell'albero
    matrice di confusione se confusion_mat = True
    ---------------------- """
    
    RF = RandomForestClassifier(criterion = criterion, max_depth = max_depth, 
          min_samples_split = min_samples_split,
          min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf,
          max_features = max_features, random_state = random,
          max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, 
          class_weight = class_weight, ccp_alpha = ccp_alpha, bootstrap = bootstrap, oob_score = oob_score,
                               warm_start = warm_start, max_samples = max_samples)
    RF.fit(X_train, Y_train)
    feature_importances_plot(model=RF, labels=feature_names)
    if verbose == 1:
        if X_val == []:
            print("L'accuracy sul test set è pari a {}".format(RF.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, RF.predict(X_test))))
        else:
            print("L'accuracy sul validation test è pari a {}".format(RF.score(X_val, Y_val)))
            print("L'accuracy sul test set è pari a {}".format(RF.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, RF.predict(X_test))))
             
    if confusion_mat == True: 
        confusion = confusion_matrix(Y_test, RF.predict(X_test))
        print("Di seguito la matrice di confusione:\n{}".format(confusion))
        return RF, confusion
    else:
        return RF
         
        
        
def adaboost(X_train, Y_train, X_test, Y_test,X_val = [], Y_val =[], random = 42,
          feature_names =  ["Non occupied","Private","Self-empinc","Self-emp-notinc","Governative","Married",
                 "Spouse absent","No_relation","Widowed","Black","Other","White",
                 "Male","United_states","Istruzione","Age_log"],
          verbose = 1,class_names = [], confusion_mat = True, base_estimator=None, n_estimators=50,
          learning_rate=1.0, algorithm='SAMME.R'):
    
    """Funzione che ritorna l'albero di classificazione e la matrice di confusione 
    
    ------------------
    Parametri:
    *: tutti i parametri dell'albero previsti da sklearn
    X_train, X_test, X_val: train, test e validation set della matrice delle X 
    Y_train, Y_test, Y_val: train, test e validation set della matrice delle Y
    verbose: Se valorizzato ad 1 fornisce informazioni degli score dei vari set
    feature_names: lista dei nomi delle categorie delle diverse features
    confusion_mat: Se True ritorna un elemento a cui è possibili assegnare la matrice di confusione
    class_names: lista delle categorie della variabile y
    
    -------------------
    Output
    fit dell'albero
    matrice di confusione se confusion_mat = True
    ---------------------- """
        
    RF = AdaBoostClassifier(base_estimator= base_estimator,
    n_estimators = n_estimators,learning_rate = learning_rate,
    algorithm= algorithm,random_state = random )
    RF.fit(X_train, Y_train)
    feature_importances_plot(model=RF, labels=feature_names)
    if verbose == 1:
        if X_val == []:
            print("L'accuracy sul test set è pari a {}".format(RF.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, RF.predict(X_test))))
        else:
            print("L'accuracy sul validation test è pari a {}".format(RF.score(X_val, Y_val)))
            print("L'accuracy sul test set è pari a {}".format(RF.score(X_test, Y_test)))
            print("Report completo sul test\n{}".format(classification_report(Y_test, RF.predict(X_test))))
        
    if confusion_mat == True: 
        confusion = confusion_matrix(Y_test, RF.predict(X_test))
        print("Di seguito la matrice di confusione:\n{}".format(confusion))
        return RF, confusion
    else:
        return RF
    
def Ada_pipeline(filepath, target, verbose = 1):
    """ Funzione che restituisce la pipeline dell'intero iter per la relizzazione 
    dell'Adaboost. Le fasi riassunte nella funzione sono le seguenti: 
    
    * 1: Divisione della X dal dataset
    * 2: Trasformazioni opportune delle variabili 
    * 3: Suddivisione del dataset in test set e train set
    * 4: Utilizzo del modello Adaboost con iperparametri ottimizzati
    -------------------------------------
    Parametri:
    filepath: filepath del dataset
    target: stringa della variabile target
    verbose: valore opzionale. Se verbose = 1 mostra tutti i passaggi effettuati
    ---------------------------------------
    Output
    modello Adaboost
    matrice di confusione 
    ----------------------------------------"""
    
    dataset = pd.read_csv(filepath).drop("Unnamed: 0", axis = 1)
    X,Y = divisione_x_y(dataset, target) 
    if verbose == 1: 
        print("Il primo passaggio prevede la suddivisione del dataset in X e Y")
        print("-" *80)
    categories = [[' No-Diploma',' HS-grad', ' Some-college',' Assoc-voc',
    ' Assoc-acdm',' Prof-school',' Bachelors',' Masters',' Doctorate']]
    X_1,ct = trasformazioni_variabili(X, ["Tipo_lavoro","Stato_sociale","Etnia","Genere", 
                                          "Paese_Nativo_cat"],["Istruzione"], categories, 
                                          [], verbose = 0)
    if verbose ==1: 
        print("""Il secondo passaggio prevede la trasformazione opportune delle diverse variabili.
In particolare\n{}""".format(ct))
        print("-"*80)
    X_train_1, X_test_1,Y_train_1,Y_test_1 = train_test_val(X_1,Y, validation= False, test_size = 0.40)
    Y_train_1,Y_test_1,le = label_encoder_test(Y_train_1,Y_test_1)
    
    if verbose == 1: 
        print("""Il terzo passaggio prevede la suddivisione del dataset in train e test.""")
        print("-"*80)
        print("""Dopo la suddivisione, utilizziamo l'Adaboost con gli iperparametri ottimizzati 
precedentemente attraverso una GridSearch cross validata""")
        print("-"*80)
        
    Ada_ottimizzato, confusion_matrix_ada = adaboost(X_train_1, Y_train_1, X_test_1, Y_test_1,
                                        base_estimator = DecisionTreeClassifier(criterion = "entropy",
                                        max_depth = 1),learning_rate= 1, n_estimators=80)
    
    return Ada_ottimizzato, confusion_matrix_ada
