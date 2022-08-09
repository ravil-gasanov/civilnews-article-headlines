import pandas as pd
import nltk
import regex
from nltk.corpus import stopwords

nltk.download('stopwords')
en_stop_words = stopwords.words('english')

def clean(raw):
    # drop a garbage column 
    raw.drop(raw.columns[-1], axis = 1, inplace = True)

    # drop rows that have missing values in either headline or views columns
    mask = raw[['headline', 'views']].dropna().index
    raw = raw.loc[mask, :]

    raw['date-time'] = pd.to_datetime(raw['date-time'], format = "%d/%m/%Y - %H:%M")
    raw.reset_index(inplace = True, drop = True)

    raw.sort_values(by = ['date-time'], ascending = True, inplace = True)
    assert raw['date-time'].is_monotonic, "not sorted"

    raw = raw.loc[pd.to_numeric(raw['views'], errors = 'coerce').dropna().index, :]
    raw['views'] = raw['views'].astype(float)

    # drop the first row with year = 1970
    raw = raw.reset_index(drop = True)
    raw = raw.loc[1:, :].reset_index(drop = True)

    return raw

def tokenize(headline):
    headline = headline.lower()
    words = regex.findall(r'\w+', headline)
    clean_words = []
  
    # what is a more idiomatic / efficient way to do this?
    for word in words:
        if word not in en_stop_words:
            clean_words.append(word)
    
    return clean_words


