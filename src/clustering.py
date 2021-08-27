import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from kmodes.kmodes import KModes
from fuzzywuzzy import fuzz
import knowledge_base
import sys 
import os

'''calcolo delle similarità tra i valori del film inserito dall utente e quelli presenti nel
cluster, restituendo il valore totale di similarità dell'intero cluster'''

def similarities(cluster,userMovie):
    totalSum = 0
    cluster['sum'] = 0
    for i in range(0,len(cluster)):
        rowSum = fuzz.ratio(cluster['genre'].values[i],userMovie['genre'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['title'].values[i],userMovie['title'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['cast'].values[i],userMovie['cast'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['director'].values[i],userMovie['director'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['year_range'].values[i],userMovie['year_range'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['description'].values[i],userMovie['description'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['country'].values[i],userMovie['country'].values[0])
        cluster['sum'].values[i] = rowSum 
        totalSum = rowSum + totalSum
    return totalSum

'''Operazioni sul dataset per rendere utilizzabile ai fine della clusterizzazione.
    Nello specifico, viene discretizzata la colonna relativa il rating e viene rimossa
    la colonna relativa la durata, poco utile ai fini del calcolo delle similarità.'''
def dataOperations(df):
    #discretizzazione colonna ratings
    bins = [0,5,np.inf]
    names = ['<5','>5']
    df['ratings_range'] = pd.cut(df['ratings'],bins, labels=names)
    df = df.drop(['ratings'],axis =1)
    df= df.dropna(subset=['ratings_range'])
    #eliminazione colonna duration
    df = df.drop(columns = ['duration'])
    return df

'''' Utilizzo del metodo del gomito per determinare il valore di k corretto.
    Queste operazioni verranno poi commentate nel main finale, poichè utilizzate durante la fase di realizzazione. '''
def KChoice(df):
    cost = []
    K = range(1,10)
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
        kmode.fit_predict(df)
        cost.append(kmode.cost_)
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()

'''Definizione di dataframe Pandas separati per ciascun cluster, rimuovendo 
    la colonna indicatrice del numero di cluster corrispondente per ciascuna row.'''
def clusterOperations(df,n):
    cluster = df[df.cluster == n]
    cluster = cluster.drop(columns = ['cluster'])
    return cluster

'''Stampa dei film da suggerire all utente.'''
def toUser(topTen):
    print('Ti consigliamo di guardare:\n')
    for element in topTen:
        print(element)

'''Definizione dei film da suggerire all utente, mediante calcoli relativi
    le similarità con i cluster.'''
def recommendation(cluster1,cluster2,cluster3,userMovie):
    totSum1 = similarities(cluster1, userMovie)
    totSum2 = similarities(cluster2, userMovie)
    totSum3 = similarities(cluster3, userMovie)
    simil = [totSum1,totSum2,totSum3]
    choice = simil.index(max(simil))
    if choice==0:
        cluster1.sort_values(by=['sum'], ascending=False, inplace = True)
        topTen = cluster1['title'].head()
    elif choice==1:
        cluster2.sort_values(by=['sum'], ascending=False, inplace = True)
        topTen = cluster2['title'].head()
    elif choice==2:
        cluster3.sort_values(by=['sum'], ascending=False, inplace = True)
        topTen = cluster3['title'].head()
    toUser(topTen)

    rispostaUtente=input('Per dettagli sulle raccomandazioni restituite, digitare kb: \n')
    if(rispostaUtente=='kb'):
        knowledge_base.explainResultsCluster(cluster1, cluster2, cluster3, simil, choice)
    else:
        print('Niente spiegazioni')


'''Classe utile per bloccare le stampe automatiche del modello.'''
class HandlePrint():
    def __init__(self):
        self.initial_stout = sys.stdout

    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def resetPrint(self):
        sys.stdout = self.initial_stout


def main(userMovie):
    df = pd.read_csv(r'..\datasets\categ_complete_dataset.csv', sep=',')
    df = dataOperations(df)
    #KChoice(df)
    
    # Building the model with 3 clusters
    handle = HandlePrint()
    handle.blockPrint()
    kmode = KModes(n_clusters=3, init = "random", n_init = 5, verbose=1)
    df['cluster'] = kmode.fit_predict(df)
    handle.resetPrint()
    cluster1 = clusterOperations(df,0)
    cluster2 = clusterOperations(df,1)
    cluster3 = clusterOperations(df,2)
    recommendation(cluster1, cluster2, cluster3, userMovie)
    
if __name__ == "__main__":
    main(sys.argv[1])
    