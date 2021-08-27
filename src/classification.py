#import
import sys
import joblib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display


from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


#funzione di conversione  dei dati per la classificazione da categorici a numerici
def prepDataset(df):
    df.drop('description', axis=1, inplace=True)

    #genre
    def creazioneArrayGenre(row, array):
        if row['genre'] is not None: 
                array.append(row['genre']);
            
    genre =[]
    df.apply(lambda row: creazioneArrayGenre(row,genre),axis=1) 
    nGenre = len(genre)
    genreDict = {}
    j = 0
    for k in range(nGenre):
        genreDict[genre[k]] = k;
        j = k
    genreDict['unknown'] = j+1
        
    def subGenre(row,dizionario):
        if row['genre'] is not None:
            element = row['genre']
            if element in dizionario:
                    row['genre'] = genreDict[element]
        return row['genre']

    df['genre']=df.apply(lambda row: subGenre(row,genreDict),axis=1)

    #type
    def creazioneArrayType(row, array):
        if row['type'] is not None: 
                array.append(row['type']);
            
    ty =[]
    df.apply(lambda row: creazioneArrayType(row,ty),axis=1) 
    nType = len(ty)
    typeDict = {}
    j = 0
    for n in range(nType):
        typeDict[ty[n]] = n;
        j = n
    typeDict['unknown'] = j+1
        
    def subType(row,dizionario):
        if row['type'] is not None:
            element = row['type']
            if element in dizionario:
                    row['type'] = typeDict[element]
        return row['type']

    df['type']=df.apply(lambda row: subType(row,typeDict),axis=1)


    #title
    def creazioneArrayTitle(row, array):
        if row['title'] is not None: 
                array.append(row['title']);
            
    title =[]
    df.apply(lambda row: creazioneArrayTitle(row,title),axis=1) 
    nTitle = len(title)
    titleDict = {}
    j = 0
    for i in range(nTitle):
        titleDict[title[i]] = i;
        j = i
    titleDict['unknown'] = j+1
        
    def subTitle(row,dizionario):
        if row['title'] is not None:
            element = row['title']
            if element in dizionario:
                    row['title'] = titleDict[element]
        return row['title']

    df['title']=df.apply(lambda row: subTitle(row,titleDict),axis=1)

    #year range
    def creazioneArrayYear(row, array):
        if row['year_range'] is not None: 
                array.append(row['year_range']);
            
    years =[]
    df.apply(lambda row: creazioneArrayYear(row,years),axis=1) 
    nYears = len(years)
    yearsDict = {}
    j = 0
    for k in range(nYears):
        yearsDict[years[k]] = k;
        j = k
    yearsDict['unknown'] = j+1
        
    def subYear(row,dizionario):
        if row['year_range'] is not None:
            element = row['year_range']
            if element in dizionario:
                    row['year_range'] = yearsDict[element]
        return row['year_range']

    df['year_range']=df.apply(lambda row: subYear(row,yearsDict),axis=1)


    #country
    def creazioneArrayCountry(row, array):
        if row['country'] is not None: 
                array.append(row['country']);
            
    country =[]
    df.apply(lambda row: creazioneArrayCountry(row,country),axis=1) 
    nCountry = len(country)
    countryDict = {}
    j = 0
    for k in range(nCountry):
        countryDict[country[k]] = k;
        j = k
    countryDict['unknown'] = j+1
        
    def subCountry(row,dizionario):
        if row['country'] is not None:
            element = row['country']
            if element in dizionario:
                    row['country'] = countryDict[element]
        return row['country']

    df['country']=df.apply(lambda row: subCountry(row,countryDict),axis=1)


    #director
    def creazioneArrayDirector(row, array):
        if row['director'] is not None: 
                array.append(row['director']);
            
    director =[]
    df.apply(lambda row: creazioneArrayDirector(row,director),axis=1) 
    nDirector = len(director)
    directorDict = {}
    j = 0
    for p in range(nDirector):
        directorDict[director[p]] = p;
        j = p
    directorDict['unknown'] = j+1
        
    def subDirector(row,dizionario):
        if row['director'] is not None:
            element = row['director']
            if element in dizionario.keys():
                row['director'] = directorDict[element]
        return row['director']

    df['director']=df.apply(lambda row: subDirector(row,directorDict),axis=1)

    #cast
    def creazioneArrayCast(row, array):
        if row['cast'] is not None: 
                array.append(row['cast']);
            
    cast =[]
    df.apply(lambda row: creazioneArrayCast(row,cast),axis=1) 
    nCast = len(cast)
    castDict = {}
    j = 0
    for p in range(nCast):
        castDict[cast[p]] = p;
        j = p
    castDict['unknown'] = j+1
        
    def subCast(row,dizionario):
        if row['cast'] is not None:
            element = row['cast']
            if element in dizionario.keys():
                row['cast'] = castDict[element]
        return row['cast']

    df['cast']=df.apply(lambda row: subCast(row,castDict),axis=1)

    #costruzione dataset definendo colonna target
    y = df['genre'] #colonna target 
    df.drop('genre', axis=1, inplace=True)
    x=df #training set

    #bilanciamento
    ros = RandomOverSampler(sampling_strategy = "not majority")
    X_res, y_res = ros.fit_resample(x,y)

    #split
    xtr,xts,ytr,yts = train_test_split(X_res,y_res,test_size=0.3,random_state=0)

    class prepElements:
        x_train = xtr
        y_train = ytr
        x_test = xts
        y_test = yts
        genreD=genreDict
        castD=castDict
        directorD=directorDict
        yearsD=yearsDict
        countryD=countryDict
        titleD=titleDict
        typeD=typeDict
        X_train_complete=X_res
        y_train_complete=y_res
    
    prep=prepElements()
    return(prep)

#funzione per la ricerca dei parametri dei classificatori
def searchClassificator(xtr,ytr,xts,yts):
    #grid searching key hyperparametres per KNeighborsClassifier
    knn = KNeighborsClassifier()

    #parametri che vogliamo testare
    n_neighbors = range(1, 21,2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'hamming']

    # definizione grid search
    scores= {'accuracy': 'accuracy','precision': make_scorer(precision_score, average ='macro'),'recall': make_scorer(recall_score, average = 'macro')}
    grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    target_names = ['anime','cult','fantasy','action','documentary','nature','romantic','sport','thrillers','kids','dramas','horror','standup','commedies','musical']

    #testing grid cv per ogni metrica in scores
    orig_stdout = sys.stdout
    writefile= open('../classification_results/knn.txt', 'a')
    sys.stdout = writefile
    for score in scores:
        print("# Tuning degli iperparametri per la metrica %s \n" % score)
        #ricerca iperparametri più performanti tramite cross validation
        grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring=score ,error_score=0)
        grid_result = grid_search.fit(xtr, ytr)
    
        print("Miglior combinazione di parametri ritrovata:\n")
        print(grid_search.best_params_)
        
        print("Classification report:\n")
        print("Il modello è stato addestrato sul training set completo\n")
        print()
        print(" Le metriche sono state calcolate sul test set.\n")
        print()
        y_true, y_pred = yts, grid_search.predict(xts)
        
        print(classification_report(y_true, y_pred,target_names=target_names))
        print()

    #chiusura file
    sys.stdout = orig_stdout
    writefile.close()


    #grid searching key hyperparametres per RandomForestClassifier
    randomForest = RandomForestClassifier()

    #parametri che vogliamo testare
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']

    # definizione grid search
    scores= {'accuracy': 'accuracy','precision': make_scorer(precision_score, average ='macro'),'recall': make_scorer(recall_score, average = 'macro')}
    target_names = ['anime','cult','fantasy','action','documentary','nature','romantic','sport','thrillers','kids','dramas','horror','standup','commedies','musical']
    grid = dict(n_estimators=n_estimators,max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


    #testing grid cv per ogni metrica in scores
    orig_stdout = sys.stdout
    writefile= open('../classification_results/randomForest.txt', 'a')
    sys.stdout = writefile
    for score in scores:
        print("# Tuning degli iperparametri per la metrica %s \n" % score)
        #ricerca iperparametri più performanti tramite cross validation
        grid_search = GridSearchCV(estimator=randomForest, param_grid=grid, n_jobs=-1, cv=cv, scoring=score ,error_score=0)
        grid_result = grid_search.fit(xtr, ytr)
    
        print("Miglior combinazione di parametri ritrovata:\n")
        print(grid_search.best_params_)
        
        print("Classification report:\n")
        print("Il modello è stato addestrato sul training set completo\n")
        print()
        print(" Le metriche sono state calcolate sul test set.\n")
        print()
        y_true, y_pred = yts, grid_search.predict(xts)
        
        print(classification_report(y_true, y_pred,target_names=target_names))
        print()

    #chiusura file
    sys.stdout = orig_stdout
    writefile.close()


    #grid searching key hyperparametres per SVC
    svc = SVC()
    #parametri da testare
    kernel = ['poly', 'rbf']
    C = [50, 10, 1.0]
    gamma = ['auto']
    probability = [True]
    scores= {'accuracy':'accuracy','precision':make_scorer(precision_score),'recall': make_scorer(recall_score)}

    # define grid search
    target_names = ['anime','cult','fantasy','action','documentary','nature','romantic','sport','thrillers','kids','dramas','horror','standup','commedies','musical']
    grid = dict(kernel=kernel,C=C,gamma=gamma, probability=probability)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    #testing grid cv per ogni metrica in scores
    orig_stdout = sys.stdout
    writefile= open('../classification_results/svc.txt', 'a')
    sys.stdout = writefile
    for score in scores:
        print("# Tuning degli iperparametri per la metrica %s \n" % score)
        if score != 'accuracy':
            grid_search = GridSearchCV(estimator=svc, param_grid=grid, n_jobs=-1, cv=cv,  scoring='%s_macro' % score,error_score=0)
        else: 
            grid_search = GridSearchCV(estimator=svc, param_grid=grid, n_jobs=-1, cv=cv,  scoring=score,error_score=0)

        grid_result = grid_search.fit(xtr, ytr)
    
        print("Miglior combinazione di parametri ritrovata:\n")
        print(grid_search.best_params_)
        print()
        
        print("Classification report:\n")
        print("Il modello è stato addestrato sul training set completo.\n")
        print("Le metriche sono state calcolate sul test set.\n")
        y_true, y_pred = yts, grid_search.predict(xts)
        
        print(classification_report(y_true, y_pred,target_names=target_names))
        print()
    
    #chiusura del file
    sys.stdout = orig_stdout
    writefile.close()


    #grid searching key hyperparameters per  BaggingClassifier
    bagging = BaggingClassifier()

    #parametri da testare
    n_estimators = [10, 100, 1000]

    # definizione grid search
    scores= {'accuracy': 'accuracy','precision': make_scorer(precision_score, average ='macro'),'recall': make_scorer(recall_score, average = 'macro')} 
    target_names = ['anime','cult','fantasy','action','documentary','nature','romantic','sport','thrillers','kids','dramas','horror','standup','commedies','musical']
    grid = dict(n_estimators=n_estimators)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    #testing grid cv for each score
    orig_stdout = sys.stdout
    writefile= open('../classification_results/bagg.txt', 'a')
    sys.stdout = writefile
    for score in scores:
        print("# Tuning iperparametri per la metrica %s \n" % score)
        #print()
        grid_search = GridSearchCV(estimator=bagging, param_grid=grid, n_jobs=-1, cv=cv, scoring=score, error_score=0)
        grid_result = grid_search.fit(xtr, ytr)
    
        print("Miglior combinazione di parametri ritrovata:\n")
        print(grid_search.best_params_)
        print()
        
        print("Classification report:\n")
        print("Il modello è stato addestrato sul training set completo.\n")
        print("Le metriche sono state calcolate sul test set.\n")
        y_true, y_pred = yts, grid_search.predict(xts)
        
        print(classification_report(y_true, y_pred,target_names=target_names))
        print()
    
    #chiusura del file
    sys.stdout = orig_stdout
    writefile.close()



#Comparazione Algoritmi 
def models_evaluation(X, y):
    # preparazione configuratione per cross validation test harness
    # preparazione modelli
    Kfold = model_selection.KFold(n_splits=10, random_state=None)
    bag_model= BaggingClassifier(n_estimators=10)
    knn_model= KNeighborsClassifier(metric='manhattan', n_neighbors= 1, weights= 'uniform')
    svc_model= SVC(C=1.0, gamma='auto', kernel= 'rbf',probability=True)
    rand_model=RandomForestClassifier(max_features = 'sqrt', n_estimators= 10)


    scoring = {'accuracy':make_scorer(accuracy_score), 
            'precision':make_scorer(precision_score, average='macro',zero_division=0),
            'recall':make_scorer(recall_score, average='macro',zero_division=0)}

    #  cross-validation su ogni classifier
    bag = cross_validate(bag_model, X, y, cv=Kfold, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=Kfold, scoring=scoring)
    knn = cross_validate(knn_model, X, y, cv=Kfold, scoring=scoring)
    rfc = cross_validate(rand_model, X, y, cv=Kfold, scoring=scoring)

    
    #crea un dataframe con i valori delle metriche
    models_scores_table = pd.DataFrame({'Bagging Classifier':[bag['test_accuracy'].mean(),
                                                              bag['test_precision'].mean(),
                                                              bag['test_recall'].mean()],
                                        
                                                        'SVC':[svc['test_accuracy'].mean(),
                                                              svc['test_precision'].mean(),
                                                              svc['test_recall'].mean()],
                                          
                                            'Random Forest':[rfc['test_accuracy'].mean(),
                                                              rfc['test_precision'].mean(),
                                                              rfc['test_recall'].mean()],
                                              
                                         'KNearestNeighbor':[knn['test_accuracy'].mean(),
                                                              knn['test_precision'].mean(),
                                                              knn['test_recall'].mean()]},
                                      
                                      index=['Accuracy', 'Precision', 'Recall'])
    
    acc = [round(bag['test_accuracy'].mean(),2),round(svc['test_accuracy'].mean(),2),round(rfc['test_accuracy'].mean(),2),round(knn['test_accuracy'].mean(),2)]
    prec = [round(bag['test_precision'].mean(),2),round(svc['test_precision'].mean(),2),round(rfc['test_precision'].mean(),2),round(knn['test_precision'].mean(),2)]
    rec = [round(bag['test_recall'].mean(),2),round(svc['test_recall'].mean(),2),round(rfc['test_recall'].mean(),2),round(knn['test_recall'].mean(),2)]

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    #tabella dei risultati
    display(models_scores_table)

    #restituisce i risultati delle metriche
    return(models_scores_table,prec,rec,acc)


#plotting dei risultati del confronto in un grafico
def plotResults(prec,rec,acc):
    #plot dei risultati della valutazione
    labels = ['Bagg', 'SVC','RandFor','KNN']

    x = np.arange(len(labels)) 
    width = 0.25 


    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width,prec, width, label='Precision', align='edge')
    rects2 = ax.bar(x, rec, width, label='Recall',  align='edge')
    rects3 = ax.bar(x + width, acc, width, label='Accuracy', align='edge')


    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric and classificator')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()


#training finale
def finalClassification(X,y):
    clf=KNeighborsClassifier(metric='manhattan', n_neighbors=1, weights= 'uniform')
    clf.fit(X,y) 
    filename ='finalized_knn.sav'
    joblib.dump(clf, filename)



#Funzione per la predizione
def predictionGenre(filename,userInput,prepElement):
    
    user=userInput.copy()
    #conversione di duration e rating a numerici
    user["duration"]=user.duration.astype(int)
    user["ratings"]=user.ratings.astype(float)
    
    #eliminazione colonne inutili per la classificazione
    user.drop('description', axis=1, inplace=True)
    user.drop('genre', axis=1, inplace=True)
    
    if user.at[0,'title'] in prepElement.titleD.keys():
        user.at[0,'title'] = prepElement.titleD[user.at[0,'title']]
    else:
        user.at[0,'title'] = prepElement.titleD['unknown']
    
    
    if user.at[0,'type'] in prepElement.typeD.keys():
        user.at[0,'type'] = prepElement.typeD[user.at[0,'type']]
    else:
        user.at[0,'type'] = prepElement.typeD['unknown']

    
    if user.at[0,'country'] in prepElement.countryD.keys():
        user.at[0,'country'] = prepElement.countryD[user.at[0,'country']]
    else:
        user.at[0,'country'] = prepElement.countryD['unknown']

    if user.at[0,'year_range'] in prepElement.yearsD.keys():
        user.at[0,'year_range'] = prepElement.yearsD[user.at[0,'year_range']]
    else:
        user.at[0,'year_range'] = prepElement.yearsD['unknown']

    if user.at[0,'cast'] in prepElement.castD.keys():
        user.at[0,'cast'] = prepElement.castD[user.at[0,'cast']]
    else:
        user.at[0,'cast'] = prepElement.castD['unknown']

    if user.at[0,'director'] in prepElement.directorD.keys():
        user.at[0,'director'] = prepElement.directorD[user.at[0,'director']]
    else:
        user.at[0,'director'] = prepElement.directorD['unknown']
    
    user['title']=user.title.astype(int)
    user['type']=user.type.astype(int)
    user['director']=user.director.astype(int)
    user['cast']=user.cast.astype(int)
    user['country']=user.country.astype(int)
    user['year_range']=user.year_range.astype(int)
    
    
    model = joblib.load(filename)
    gen= model.predict(user)
    
    for genre, val in prepElement.genreD.items(): 
      if val == gen:
          gen=genre
    
    userInput.at[0,'genre'] = gen
    return gen 
    


def main(userMovie):
    
    df = pd.read_csv(r'..\datasets\categ_complete_dataset.csv',sep=',')
        
    prepInfo=prepDataset(df)

    #ricerca parametri classificatori 
    #searchClassificator(xtr,ytr,xts,yts)
        
    #confronto classificatori
    #model, prec, rec, acc = models_evaluation(prepInfo.X_train_complete,prepInfo.y_train_complete)#verifica se devi usare xtr,ytr
    
    #plotting dei risultati in un grafico 
    #plotResults(prec,rec,acc)
    
    #Fitting con il classificatore corretto
    #finalClassification(prepInfo.X_train_complete,prepInfo.y_train_complete)
    
    #predizione del genere del file dato dall'utente
    result= predictionGenre(r"finalized_knn.sav",userMovie,prepInfo)#user movie dovrebbe essere globale nel main
    print('Il genere del film o serie TV da te inserito è %s \n' % result)

       

if __name__ == "__main__":
    main(sys.argv[1])
