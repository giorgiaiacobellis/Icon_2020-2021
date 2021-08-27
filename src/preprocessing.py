import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

#unificazione delle colonne della durata
def durationUnification(row):
    if pd.isna(row['duration_x']) :
        return row['duration_y']
    else :
        return row['duration_x']

#unificazione sulla colonna dei registi
def directorUnification(row):
    if pd.isna(row['director_x']) :
        return row['director_y']
    else :
        return row['director_x']

#unificazione sulla colonna del cast
def castUnification(row):
    if pd.isna(row['cast_x']) :
        return row['cast_y']
    else :
        return row['cast_x']

#unificazione sulla colonna dell'anno di rilascio
def yearUnification(row):
    if pd.isna(row['release_year_x']) :
        return row['release_year_y']
    else :
        return row['release_year_x']

#unificazione sulla colonna del genere 
def genreUnification(row):
    if pd.isna(row['genre_x']) :
        return row['genre_y']
    else :
        return row['genre_x']

#unificazione sulla colonna del paese
def countryUnification(row):
    if pd.isna(row['country_x']) :
        return row['country_y']
    else :
        return row['country_x']

#unificazione sulla colonna del rating
def ratingUnification(row):
    if pd.isna(row['rating']) :
        return row['mean_vote']
    else :
        return row['rating']

def unificationColumns(movieData):
    movieData['duration']= movieData.apply(lambda row: durationUnification(row),axis=1) 
    movieData['director']= movieData.apply(lambda row: directorUnification(row),axis=1) 
    movieData['cast']= movieData.apply(lambda row: castUnification(row),axis=1) 
    movieData['release_year']= movieData.apply(lambda row: yearUnification(row),axis=1) 
    movieData['genre']= movieData.apply(lambda row: genreUnification(row),axis=1) 
    movieData['country']= movieData.apply(lambda row: countryUnification(row),axis=1) 
    movieData = movieData.drop(columns = ['duration_x','duration_y','director_x','director_y','cast_x','cast_y','release_year_x','release_year_y', 'genre_x','genre_y','country_x', 'country_y'])
    return movieData

#conversione variabili categoriche genere in variabili numeriche usando le dummy variables
def split_genres(row):
    if not(pd.isna(row['genre'])) :
        actors = row['genre'].split(', ')
        return actors

def transformationGenre(element,array,new):
    if element in array:
        element = new
    return element


def genericWords(analize, replace):
    returning = analize
    if (analize.find(replace) != -1):
        returning = analize.replace(replace, '2')
    return returning

def deleteElement(element, array,row):
   if element in array:
       return 1
   return row 
    
#creazione dictionary generi per corrispondenza con valore numerico assegnato
def createArrayGenres(row, genre):
    if row['genre'] is not None: 
        for element in row['genre']:
            genre.append(str(element))
      
#trasformazione da unico elemento dell'array row a elemento a se stante
def daArray(row):
    a = []
    if row['genre'] != a:
        return row['genre'][0]
    return None
    
def replaceGenre(row,dictGnr):
    if row['genre'] is not None:
        element = row['genre']
        if element in dictGnr:
            row['genre'] = dictGnr[element]
    return row['genre']

#operazioni su colonna country

#mantenimento unico paese per film
def split_country(row):
    if not(pd.isna(row['country'])):
        countries = row['country'].split(',')
        return countries[0] 

#operazioni su colonna cast

#mantenimento un attore per film
def split_cast(row):
    if not(pd.isna(row['cast'])):
        actors = row['cast'].split(',')
        return actors[0] 

#da categoriche a numeriche per il cast

def createArray(row, cast):
    if row['cast'] is not None: 
        cast.append(row['cast'])
            
def replaceActors(row,dictionary):
    if row['cast'] is not None:
        element = row['cast']
        if element in dictionary:
            row['cast'] = dictionary[element]
    return row['cast']

def inverseActors(row,dictionary):
    if row['cast'] is not None:
        element = row['cast']
        for categorical, numerical in dictionary.items():
            if numerical == element:
                row['cast'] = categorical
    return row['cast']

def inverseGenre(row,dictionary):
    if row['genre'] is not None:
        element = row['genre']
        for categorical, numerical in dictionary.items():
            if numerical == element:
                row['genre'] = categorical
    return row['genre']

def main():
    #lettura e creazione dataframe dei due dataset
    movieData1 = pd.read_csv(r'..\datasets\1_Netflix_Movie.csv', sep=',')
    movieData2 = pd.read_csv(r'..\datasets\2_Netflix_Movie.csv', sep=',')
    movieRatings = pd.read_csv(r'..\datasets\IMDb_rating.csv', sep=',')
    #eliminazione features superflue
    movieData1 = movieData1.drop(columns = ['rating','date_added', 'id'])
    movieData2 = movieData2.drop(columns = ['enter_in_netflix', 'id'])
    #modifica del tipo di duration per unificazione
    movieData2=movieData2.astype({"Duration": object})
    #rinominazione colonne per concordanza 
    movieData2 = movieData2.rename(columns= {"movie_name": "title", "Duration":"duration","year":"release_year", "actors":"cast"})
    movieData1 = movieData1.rename(columns = {"listed_in":"genre"})
    movieData = pd.merge(movieData1,movieData2 ,on=["title"] , how="outer" )
    #unificazioni colonne appartenenti a dataset diversi
    movieData = unificationColumns(movieData)
    #rimuoviamo le descrizioni nulle perchè non è possibile rimediarvi con la
    #feature imputation 
    movieData= movieData.dropna(subset=['description'])
    descr= movieData.loc[:,'description'].values
    #assegnazione del valore Movie ai film del secondo dataset    
    movieData['type']=movieData['type'].replace({np.nan: 'Movie'})    
    movieData = pd.merge(movieData,movieRatings ,on=["title"] , how="left" )
    movieData['ratings']= movieData.apply(lambda row: ratingUnification(row),axis=1)
    movieData = movieData.drop(columns = ['rating','mean_vote','Unnamed: 0','total_votes'])
    #rimozione duplicati
    movieData = movieData.drop_duplicates(subset=['title'])
    #discretizzazione valori della colonna 'year'
    bins = [1900,1950,1960,1970,1980,1990,1995,2000,2005,2010,2015,2020,np.inf]
    names = ['<1950','1950-1960','1960-1970','1970-1980','1980-1990','1990-1995','1995-2000','2000-2005','2005-2010','2010-2015','2015-2020', '>2020']
    movieData['year_range'] = pd.cut(movieData['release_year'],bins, labels=names)
    movieData = movieData.drop(['release_year'],axis =1)
    #normalizzazione valori di 'duration'
    movieData['duration'] = movieData['duration'].str.extract('(\d+)',expand=False)
    movieData= movieData.dropna(subset=['duration'])
    movieData['duration'] = movieData['duration'].astype(int)
    movieData['genre']= movieData.apply(lambda row: split_genres(row),axis=1) 
    for row in movieData['genre']:
        if row is not None:
            index = 0
            for element in row:
                row[index] = element.strip()
            index = index + 1
    #COMEDIES per '2 Comedies', Comedies
    #HORROR per '2 Horror', 'Horror 2'
    #DRAMAS per '2 Dramas', 'Dramas'
    #ROMANTIC per 'Romantic 2'
    #ANIME per 'Anime Features', 'Anime Series'
    #CULT per 'Classic & Cult 2', 'Classic 2', 'Cult 2'
    #THRILLERS per '2 Thrillers', '2 Mysteries', 'Thrillers'
    #KIDS per 'Children & Family 2', 'Kids' 2 ' , 'Teen 2'
    #FANTASY per '2 Sci-Fi & Fantasy', 'Sci-Fi & Fantasy'
    #NATURE per 'Science & Nature 2', 'Faith & Spirituality'
    #STANDUP per 'Stand-Up Comedy', 'Stand-Up Comedy & Talk 2'
    #DOCUMENTARY per 'Documentaries','Docuseries'
    #ACTION per '2 Action & Adventure", 'Action & Adventure', 'Crime 2'
    #MUSICAL per 'Music & Musical'
    #SPORT per 'Sports 2'
    #conversione dei generi in una versione standardizzata adottata nel codice 
    commedies = ['2 Comedies', 'Comedies']
    horror = ['Horror 2', '2 Horror']
    dramas = ['2 Dramas', 'Dramas']
    romantic = ['Romantic 2']
    anime = ['Anime Features', 'Anime Series']
    cult = ['Classic & Cult 2', 'Classic 2', 'Cult 2']
    thrillers = ['2 Thrillers', '2 Mysteries', 'Thrillers'] 
    kids = ['Children & Family 2', "Kids\' 2" , 'Teen 2']
    fantasy =['2 Sci-Fi & Fantasy', 'Sci-Fi & Fantasy'] 
    nature =['Science & Nature 2', 'Faith & Spirituality']
    standup =  ['Stand-Up Comedy', 'Stand-Up Comedy & Talk 2','Stand-Up Comedy & Talk Shows']
    documentary = ['Documentaries','Docuseries'] 
    action = ['2 Action & Adventure', 'Action & Adventure', 'Crime 2']
    musical = ['Music & Musicals']
    sport = ['Sports 2']
    remove = ['LGBTQ', 'Independent 2', 'International 2', 'Korean 2', 'Reality 2', 'Spanish-Language 2', '2', 'LGBTQ 2']
    for row in movieData['genre']:
        if row is not None:
            index = 0
            for element in row:
                row[index]= genericWords(row[index],'TV Shows')
                row[index]= genericWords(row[index],'TV Show')
                row[index]= genericWords(row[index],'Movies')
                row[index]= genericWords(row[index],'Movie')
                row[index]= genericWords(row[index],'TV')
                row[index] = transformationGenre(row[index], commedies, 'commedies')
                row[index] = transformationGenre(row[index], horror, 'horror')
                row[index] = transformationGenre( row[index], dramas, 'dramas')
                row[index] = transformationGenre( row[index], romantic, 'romantic')
                row[index] = transformationGenre( row[index], anime, 'anime')
                row[index] = transformationGenre( row[index], cult, 'cult')
                row[index] = transformationGenre( row[index], thrillers, 'thrillers')
                row[index] = transformationGenre( row[index], kids, 'kids')
                row[index] = transformationGenre( row[index], fantasy, 'fantasy')
                row[index] = transformationGenre( row[index], nature, 'nature')
                row[index] = transformationGenre( row[index], standup, 'standup')
                row[index] = transformationGenre( row[index], documentary, 'documentary')
                row[index] = transformationGenre( row[index], action, 'action')
                row[index] = transformationGenre( row[index], musical, 'musical')
                row[index] = transformationGenre( row[index], sport, 'sport')
                row[index] = transformationGenre(row[index],remove, 'other')
                index=index+1       
    for row in movieData['genre']:
        if row is not None:
            row[:] = filter(lambda a: a != 'other', row)
            row[:] = list(dict.fromkeys(row))
    genres =[]
    movieData.apply(lambda row: createArrayGenres(row,genres),axis=1) 
    nGenres = len(genres)
    dictGenre = {}
    for i in range(nGenres):
        dictGenre[str(genres[i])] = i

    genres = list(set(genres))
    #occurrences generi
    occGnr = {}
    for key in dictGenre:
        cont = 0
        for row in movieData['genre']:
            i=0
            if row is not None:
                for element in row:
                    if key == row[i]:
                        cont = cont + 1
                    i = i+1
        occGnr[key] = cont
    #riduzione ad unico genere
    for row in movieData['genre']:
        if row is not None:
            i=0
            occ= 0
            ind=0
            for element in row:
                if occGnr[row[i]]>occ :
                    occ= occGnr[row[i]]
                    ind=i
                i=i+1
            row[:] = filter(lambda a: a == row[ind], row)
    movieData['genre']= movieData.apply(lambda row: daArray(row),axis=1)
    movieData['genre']=movieData.apply(lambda row: replaceGenre(row,dictGenre),axis=1)
    #da categoriche a numeriche per il genere 
    #movieData['genre'] = labelEncoder.fit_transform(movieData['genre'])
    #da categoriche a numeriche per il type 
    labelEncoderType = LabelEncoder()
    movieData['type'] = labelEncoderType.fit_transform(movieData['type'])
    #da catecorighe a numeriche per il director
    labelEncoderDirector = LabelEncoder()
    movieData['director'] = labelEncoderDirector.fit_transform(movieData['director'])
    #da categoriche a numeriche per il release_year 
    labelEncoderYear = LabelEncoder()
    movieData['year_range'] = labelEncoderYear.fit_transform(movieData['year_range'])
    movieData['country']= movieData.apply(lambda row: split_country(row),axis=1) 
    #da categoriche a numeriche per country
    labelEncoderCountry = LabelEncoder()
    movieData['country'] = labelEncoderCountry.fit_transform(movieData['country'])
    #da categoriche a numeriche per il titolo
    labelEncoderTitle = LabelEncoder()
    movieData['title'] = labelEncoderTitle.fit_transform(movieData['title'])
    movieData['cast']= movieData.apply(lambda row: split_cast(row),axis=1) 
    #eliminare spazi inizio e fine di ciascun attore, per evitare che vi siano ridondanze nei nomi
    for row in movieData['cast']:
        if row is not None:
                row= row.strip()
    cast = []
    movieData.apply(lambda row: createArray(row,cast),axis=1) 
    nCast = len(cast)
    dictionary = {}
    for i in range(nCast):
        dictionary[cast[i]] = i
    movieData['cast']=movieData.apply(lambda row: replaceActors(row,dictionary),axis=1)
    occurrences = {}
    for key in dictionary:
        cont = 0
        for row in movieData['cast']:
            if row is not None:
                if dictionary[key] == row:
                    cont = cont + 1
        occurrences[key] = cont
    descr = movieData['description']
    #rimozione descr
    movieData = movieData.drop(columns = ['description'])
    #imputation per generi effettuata con "hot-deck imputation" che consiste nello scegliere casualmente il
    #valore mancante da un insieme di variabili simili
    movieData['genre'].fillna(method='ffill', inplace=True)
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    movieData['genre'] = imputer.fit_transform(movieData[['genre']])
    #cancellazione film senza cast
    movieData.dropna(subset=['cast'],inplace = True,how='all')
    #feature imputation con KNN 
    movies = movieData.iloc[:,[1,3,4,5,7]].to_numpy()
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    movies= imputer.fit_transform(movies)
    pdMovies= pd.DataFrame(movies,columns=(['title','director','cast','genre','ratings']))
    pdMovies = pdMovies.drop(columns=['director','cast','genre'])
    movieData = pd.merge(movieData, pdMovies, on=["title"], how='left')
    movieData = movieData.drop(columns=['ratings_x'])
    movieData = movieData.rename(columns = {"ratings_y":"ratings"})
    columns = ['type','title','duration','director','cast','genre', 'country', 'year_range', 'ratings','description']
    df = pd.DataFrame(columns = columns)
    df = pd.merge(df, descr, on=['description'], how='left')
    df['type'] = labelEncoderType.inverse_transform(movieData['type'])
    df['title'] = labelEncoderTitle.inverse_transform(movieData['title'])   
    df['director'] = labelEncoderDirector.inverse_transform(movieData['director'])
    df['country'] = labelEncoderCountry.inverse_transform(movieData['country'])
    df['year_range'] = labelEncoderYear.inverse_transform(movieData['year_range'])
    df['duration'] = movieData['duration']
    df['description'] = descr
    df['cast']=movieData.apply(lambda row: inverseActors(row,dictionary),axis=1)
    df['genre']=movieData.apply(lambda row: inverseGenre(row,dictGenre),axis=1)
    df['ratings'] = movieData['ratings']
    df= df.dropna(subset=['director','country','description'])
    df.to_csv(r'..\datasets\categ_complete_dataset.csv', index = False)
    return df

if __name__ == "__main__":
    df = main()