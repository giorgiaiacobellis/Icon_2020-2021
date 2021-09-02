import pandas as pd
import numpy as np
# Inizio KB
# Lettura csv
movieDataString = pd.read_csv(r'..\datasets\categ_complete_dataset.csv', sep=',')

#Creazione delle liste per ogni singola colonna
type = movieDataString.loc[:,'type']
title = movieDataString.loc[:, 'title']
duration = movieDataString.loc[:, 'duration']
director = movieDataString.loc[:, 'director']
genre = movieDataString.loc[:, 'genre']
country = movieDataString.loc[:, 'country']
ratings = movieDataString.loc[:, 'ratings']
year_range= movieDataString.loc[:, 'year_range']

# dato un genere, restituisce una lista con tutti i film che trova di quel genere
def trovaGenere(genere:str):    
    j = 0
    listaRisultati = {}
    for i in range(len(genre)):
        if(genre[i] == genere):
            listaRisultati[j] = [title[i]]
            j += 1         
    return listaRisultati

# determina se un film esiste
def titoloEsiste(titolo:str):
    for i in range(len(title)):
        if(title[i] == titolo or title[i].lower() == titolo):
            return True
    return False

# determina in che posizione si trova il film
def whereTitoloEsiste(titolo:str):
    for i in range(len(title)):
        if(title[i] == titolo or title[i].lower() == titolo):
            return i
    return i # altrimenti ritonra un mess di errore

# determina se un determinato genere esiste nel dataset
def genereEsiste(genere:str):
    for i in range(len(genre)):
        if(genre[i] == genere):
            return True
    return False

# dato un indice, restituisce il genere corrispondente a quell'indice
def estrapolaGenere(i: int):
    return genre[i]

# determina se i due generi in input sono uguali
def generiUguali(primoGenere:str, secondoGenere:str):
    if(primoGenere==secondoGenere):
        return True
    return False

# trova la corrispondenza nel dataset tra un titolo e un genere
def corrispondenzaEsiste(titolo:str, genere:str):
    i = whereTitoloEsiste(titolo)

    if(genre[i]==genere):
        return True
    return False

# dato un titolo e un genere in input, risponde se ha trovato una corrispondenza nel dataset
def askGenereDaTitolo(titolo:str,genere:str):
    # Controllo se titolo scritto bene
    if(not titoloEsiste(titolo)):
        print("Il titolo inserito non esiste") 
        return

    #controllo se genere scritto bene
    if(not genereEsiste(genere)):
        print("Hai inserito un genere non presente") 
        return
    
    stringa = titolo + "_" + genere
    
    # Trova la corrispondenza tra titolo e genere
    risposta = corrispondenzaEsiste(titolo,genere) 
    if(risposta):
        print("YES")
    else:
        print("NO")
    
    # spiega come si è arrivati alla soluzione
    rispostaUtente=input("Digitare how per la spiegazione: ")
    if (rispostaUtente.lower()=="how"):
       print("askGenereDaTitolo("+titolo+","+genere+") <=> "+stringa)
       rispostaUtente=input("Digitare 'how i' specificando al posto di i il numero dell'atomo : ")
       if(rispostaUtente.lower()=='how 1'):
           print(stringa + " <=>", risposta)
       else:
           print("Errore di digitazione")
    else:
         print("Errore di digitazione")
    
# dati due titoli, risponde se presentano lo stesso genere 
def askStessoGenere(titolo1:str, titolo2:str):
    
    if(not titoloEsiste(titolo1)):
        print("Il primo titolo inserito non è presente")
        return
    if(not titoloEsiste(titolo2)):
        print("Il secondo titolo inserito non è presente")
        return
    
    # Identifico la posizione dei titoli visto che sono presenti nel dataset
    primoTitolo = whereTitoloEsiste(titolo1)
    secondoTitolo = whereTitoloEsiste(titolo2)

    # Estrapolo i generi dei titoli
    primoGenere = estrapolaGenere(primoTitolo)
    secondoGenere = estrapolaGenere(secondoTitolo)
               
    risposte = {}
    
    risposte[1] = corrispondenzaEsiste(titolo1,primoGenere)
    risposte[2] = corrispondenzaEsiste(titolo2, secondoGenere)
    risposte[3] = generiUguali(primoGenere, secondoGenere)

    if(risposte.get(1) == True and risposte.get(2) == True and risposte.get(3) == True):
        print("YES")
    else:
        print("NO")
    
    # Spiega come si è arrivati ai risultati
    rispostaUtente=input("Digitare how per la spiegazione: ")
    if (rispostaUtente.lower()=="how"):
       print("askStessoGenere("+titolo1+","+titolo2+") <=> "+titolo1+"_"+primoGenere+ " and "+titolo2+"_"+secondoGenere+" and generiUguali("+primoGenere+","+secondoGenere+")")

       rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
       if(rispostaUtente.lower()=='how 1'):
           print(titolo1+"_"+primoGenere+" <=>", risposte.get(1))
           rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
           if(rispostaUtente.lower() =="how 2"):
               print(titolo2+"_"+secondoGenere+" <=>",risposte.get(2))       
               rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
               if(rispostaUtente=="how 3"): 
                   print("generiUguali("+primoGenere+","+secondoGenere+") <=>", risposte[3])       
               else:
                   print("Errore di digitazione")
           else: 
               print("Errore di digitazione")
       else:
           if(rispostaUtente.lower() =="how 2"):
               print("SecondaCorrispondenza("+titolo2+") <=> corrispondenzaEsiste("+titolo2+" , " + secondoGenere+ ") <=>", risposte.get(2))       
               rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
               
               if(rispostaUtente.lower()=="how 1"):
                   print("PrimaCorrispondenza("+titolo1+") <=> corrispondenzaEsiste("+titolo1+" , " + primoGenere+ ") <=>", risposte.get(1))
               else:
                   if(rispostaUtente=="how 3"): 
                       print("stessoGenere("+titolo1+","+titolo2+") <=> "+primoGenere+"_"+secondoGenere+" <=>", risposte[3])       
                   else:
                       print("Errore di digitazione")
                
           else:
               if(rispostaUtente.lower() =="how 3"):
                   print("stessoGenere("+titolo1+","+titolo2+") <=>  generiUguali("+primoGenere+","+secondoGenere+") <=>", risposte[3])
                   rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
                   if(rispostaUtente.lower() =="how 2"):
                       print("SecondaCorrispondenza("+titolo2+") <=> corrispondenzaEsiste("+titolo2+" , " + secondoGenere+ ") <=>", risposte.get(2))       
                       rispostaUtente=input("Digitare 'how i' specificando in i il numero dell'atomo per ulteriori informazioni: ")
                        
                       if(rispostaUtente.lower()=="how 1"):
                           print("PrimaCorrispondenza("+titolo1+" ) <=> corrispondenzaEsiste("+titolo1+" , " + primoGenere+ ") is ", risposte.get(1))
                       else:
                           if(rispostaUtente=="how 2"): 
                               print("SecondaCorrispondenza("+titolo2+") <=> corrispondenzaEsiste("+titolo2+" , " + secondoGenere+ ") is ", risposte.get(2))       
                           else:
                               print("Errore di digitazione")
               else:
                   print("Errore di digitazione")
    else:
        print("Errore di digitazione") 

# spiega i risultati ottenuti dalle raccomandazioni effettuate col clustering
def explainResultsCluster(cluster1, cluster2, cluster3, similarities, choice):
    choice+=1
    print('Il cluster di appartenenza è il valore di choice:', choice)
    print('Le metriche restituite tra tutti i cluster sono le seguenti:', similarities )
    if(choice == 1):
        copia = cluster1.drop(columns=['ratings_range','type','genre','cast', 'director','year_range','country','description'])
        copia = copia.rename(columns={"sum": "similarity"})
        print('\nLe singole metriche di similarità restituite per il cluster', choice, 'sono:\n', copia.head(), '\n')
    if(choice == 2):
        copia = cluster2.drop(columns=['ratings_range','genre','cast','director','year_range','country','description'])
        copia = copia.rename(columns={"sum": "similarity"})
        print('\nLe singole metriche di similarità restituite per il cluster', choice, 'sono:\n', copia.head(), '\n')
    if(choice == 3):
        copia = cluster3.drop(columns=['ratings_range','genre','cast','director','year_range','country','description'])
        copia = copia.rename(columns={"sum": "similarity"})
        print('\nLe singole metriche di similarità restituite per il cluster', choice ,'sono: \n', copia.head(), '\n')

def mainFunz():
    # askGenereDaTitolo
    print('1) Dato un titolo e un genere in input, la KB è in grado di dirti se il titolo corrisponde al genere indicato grazie alla funzione askGenereDaTitolo, rispondendo YES se effettivamente corrisponde, altrimenti NO \n')
    
    # askStessoGenere
    print('2) Dati due titoli in input, la KB è in grado di dirti se il genere dei due film è lo stesso oppure no grazie alla funzione askStessoGenere, rispondendo YES se corrispondono, NO altrimenti\n')
    
    rispostaUtente=input("Digitare il numero della funzione che si vuole eseguire : ")
    if (rispostaUtente=="1"):
        titoloUtente = input("Digitare il titolo del film: ")
        genereUtente = input("Digitare il genere del film: ")
        askGenereDaTitolo(titoloUtente, genereUtente)   
    else:
        if(rispostaUtente=="2"):
            titoloUtente1 = input("Digitare il titolo del primo film: ")
            titoloUtente2 = input("Digitare il titolo del secondo film: ")
            askStessoGenere(titoloUtente1,titoloUtente2) 
        else: 
            print('Errore di digitazione')

if __name__ == "__main__":
    mainFunz()