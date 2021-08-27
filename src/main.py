import pandas as pd
import string
import classification as clf
import clustering as clust
import preprocessing as prep
import knowledge_base as kb


#funzione per l'inserimento dei dati del film dall'utente
def getUserMovie(choice):   
    print('[NB: inserire i dati dei film rispettando la dicitura ufficiale] \n(es. Avengers: Infinity War-> OK  ma avengers infinity war->NO) \n')
    genreM=''
    title = input("Inserire il nome del film o serie TV:")
    typeM = input(title + ' è un film? (s/n) \n -> ')
    if typeM == 's':
        typeM = 'Movie'
    else:
        typeM = 'TV Show'
    country = input('Inserire il paese di produzione: \n -> ')
    yr = input ('Inserire l`anno di rilascio: \n -> ')
    yr=releaseYear(yr)
    direc = input ('Inserire il regista: \n -> ')
    cast = input('Inserire un membro del cast: \n -> ')
    description = input('Inserire parole chiave in inglese su film/serie TV: \n -> ')
    rating = input('Inserire un voto da 1 a 10 sul film/serie TV: \n -> ')
    duration =input('Inserire la durata di film/serie TV: \n -> ')
    if choice == 2:
        genreM = input('Inserisci il genere, scegliendo tra questi:\n1 action \n2 anime \n3 commedies \n4 cult \n5 documentary \n6 dramas \n7 fantasy \n8 horror \n9 kids \n10 musical \n11 nature \n12 romantic \n13 sport \n14 stand-up \n15 thrillers\n ->')
        if (genreM == '1'):
            genreM = 'action'
        elif (genreM == '2'):
            genreM = 'anime'
        elif (genreM == '3'):
            genreM = 'comedies'
        elif (genreM == '4'):
            genreM = 'cult'
        elif (genreM == '5'):
            genreM = 'documentary'
        elif (genreM == '6'):
            genreM = 'dramas'
        elif (genreM == '7'):
            genreM = 'fantasy'
        elif (genreM == '8'):
            genreM = 'horror'
        elif (genreM == '9'):
            genreM = 'kids'
        elif (genreM == '10'):
            genreM = 'musical'
        elif (genreM == '11'):
            genreM = 'nature'
        elif (genreM == '12'):
            genreM = 'romantic'
        elif (genreM == '13'):
            genreM = 'sport'
        elif (genreM == '14'):
            genreM = 'standup'
        elif (genreM == '15'):
            genreM = 'thrillers' 
        else:
            while (not 1 <= int(genreM) <= 15):
                genreM = input("Perfavore, inserisci un numero corretto.\n") 
    data = {'type':[typeM], 'title':[title],'duration':[duration],'director':[direc],'cast':[cast],'country':[country], 'year_range':[yr], 'ratings':[rating],'genre': [genreM],'description':[description]}
    userMovieDF = pd.DataFrame(data)
    return userMovieDF


#funzione del menu principale
def menu():
    print("Benvenuto in MovieLand!")
    choice = input("Scegli come proseguire: \n 1. Scopri il genere di un film o serie TV\n 2. Lasciati suggerire un nuovo film sulla base di un altro che hai apprezzato \n --> ")
    while (int(choice) != 1 and int(choice) != 2):
        choice = input("Inserisci un'opzione valida --> ")
    return int(choice)

#funzione calcolo range anno di rilascio
def releaseYear(year):
    yearI=int(year)
    
    if (yearI < 1950):
            year = '<1950'
    elif (yearI >= 1950 and yearI < 1960):
            year ='1950-1960'
    elif (yearI >= 1960 and yearI < 1970):
            year ='1960-1970'
    elif (yearI >= 1970 and yearI < 1980):
            year ='1970-1980'
    elif (yearI >= 1980 and yearI < 1990):
            year ='1980-1990'
    elif (yearI >= 1990 and yearI < 1995):
            year ='1990-1995'
    elif (yearI >= 1995 and yearI < 2000):
            year ='1995-2000'
    elif (yearI >= 2000 and yearI < 2005):
            year ='2000-2005'
    elif (yearI >= 2005 and yearI < 2010):
            year ='2005-2010'
    elif (yearI >= 2010 and yearI < 2015):
            year ='2010-2015'
    elif (yearI >= 2015 and yearI < 2020):
            year ='2015-2020'
    return year

def main():
    #preprocessing dei dati
    #prep.main()
    
    #menu
    choice = menu()
    print('INIZIAMO! \n')
    userMovie = getUserMovie(choice)
    if choice == 1:
        #classificazione genere
        clf.main(userMovie)
    else:
        #recommender system
        clust.main(userMovie)
        
    #interrogazione della base di conoscenza sul film
    print('INTERROGA IL SISTEMA!\n')
    kb.mainFunz()
    
if __name__ == "__main__":
    main()
