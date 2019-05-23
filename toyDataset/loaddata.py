import pandas as pd
from os import path

path100k = path.dirname(__file__) + r'\1K'

def load100KRatings():
    df = pd.read_table(path100k+r'\u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    return df

def load100KItemSide():
    import codecs
    with codecs.open(path100k+'/u.item', 'r', 'utf-8', errors='ignore') as f:
        movies = pd.read_table(f, delimiter='|', header=None,names="itemId| movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western ".split('|'))
    return movies

def load100kUserSide():
    import codecs
    with codecs.open(path100k + '/u.user', 'r', 'utf-8', errors='ignore') as f:
        users = pd.read_table(f, delimiter='|', header=None,names="userId| age | gender | occupation | zip code".split('|'))
    return users