import pandas as pd
import nltk
import regex
from nltk.corpus import stopwords

nltk.download('stopwords')
en_stop_words = stopwords.words('english')

def tokenize(headline):
    headline = headline.lower()
    words = regex.findall(r'\w+', headline)
    clean_words = []
  
    # what is a more idiomatic / efficient way to do this?
    for word in words:
        if word not in en_stop_words:
            clean_words.append(word)
    
    return clean_words