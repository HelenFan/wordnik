from flask import (
    Flask, 
    request,
    jsonify,
)
from pathlib import Path
import os
import numpy as np
import json
import pandas as pd
import scipy
from gensim.models import KeyedVectors
import pickle
from pathlib import Path
from nltk.stem.porter import PorterStemmer
import logging
from google.cloud import storage

app = Flask(__name__)
DATA_PATH = Path('/home/gcmac/prod/prod_data')


def get_clean_data(path):
    df = pd.read_json(path)
    for c in df.columns:
        df[c] = df[c].apply(lambda x: list(x.values())[0] if type(x) == dict else x)
    return df

def get_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data


def filter_words(input_words, recommeded_words, stemmer):
    """Filter out words that have the same stem
    
    Parameters
    ----------
    input_words : list
        list of input words
        
    recommeded_words : list
        list of recommended words

    stemmer : nltk.stem.porter.PorterStemmer
        function that converts full word to stem of word
    
    Returns
    -------
    out : list
        list of trimmed down version of lst that does not contain words that share same stems with input_
    """
    input_stems = set([stemmer.stem(input_word) for input_word in input_words])
    return [word for word in recommeded_words if stemmer.stem(word) not in input_stems]


def find_rank(words, popularity_metric):
    """Given a list of (recommended) words, rank them based on popularity.
    
    Parameters
    ----------
    words : list
        list of words
    popularity_metric : dictionary
        this is the "popularity dictionary", which shows the number of times each word is searched/tagged/listed
    
    Returns
    -------
    df : pandas.DataFrame
        table containing the original list of words, and their popularity ranking
    """
    words_in_pop_list = [word for word in words if word in popularity_metric]
    words_not_in_pop_list = [word for word in words if word not in popularity_metric]
    df_in_list = pd.DataFrame(
        [popularity_metric[word] for word in words_in_pop_list],
        index=words_in_pop_list,
        columns=['count']
    )

    df_not_in_list = pd.DataFrame(
        [0]*len(words_not_in_pop_list),
        index=words_not_in_pop_list,
        columns=['count']
    )

    df = pd.concat([df_in_list, df_not_in_list])
    df['rank'] = df['count'].rank(ascending=False)
    return df


def find_words(input_words, n_outputs=1):
    """Given an input list of words, find recommended words based on similarities.
    
    Parameters
    ----------
    input_words : list
        list of words that the recommendation will be based on
    
    n_outputs: integer
        the number of recommended (most similar) words wanted
    
    Returns
    -------
    out : list
        list of words recommended
    
    """
    most_similar = model.most_similar(
        [word for word in input_words if word in model.vocab],
        topn=n_outputs
    )

    results = []
    for word, pred in most_similar:
        if word.isalpha() and word not in input_words:
            results.append(word)
        if len(results) >= n_outputs:
            return results
    return results


def recommend_words(input_words, n_outputs=5, multiple=5):
    stemmer = PorterStemmer()
    candidates = find_words(input_words, n_outputs=n_outputs*multiple)
    filtered_candidates = filter_words(input_words, candidates, stemmer)
    
    rank_df = find_rank(filtered_candidates, list_popularity).reset_index()
    rank_df.rename(columns={'index':'word'}, inplace=True)
    rank_df['stem'] = rank_df['word'].apply(lambda x: stemmer.stem(x))
    rank_df.drop_duplicates(['stem'],inplace=True)
    
    return list(rank_df.sort_values('rank')['word'].values[:n_outputs])


@app.before_first_request
def load_data():
    global listed_words
    global list_popularity
    global word_lists
    global list_vecs
    listed_words = get_clean_data(DATA_PATH/'valid_listed_words')
    list_popularity = dict(listed_words.lcword.value_counts())
    word_lists = get_clean_data(DATA_PATH'/valid_list_metadata')
    list_vecs = get_pickle(DATA_PATH/'listvecs.pickle')


    global model
    model_path = DATA_PATH/'wiki.en.vec'
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)


@app.route('/test_server')
def hello_world():
    return "Yep - the server's lighs are on and you know the address"


@app.route('/word_pred/', methods=['POST'])
def word_pred():
    data = request.get_json()
    input_words = data['input_words']
    n_outputs = data.get('n_outputs') or 10
    multiple = data.get('multiple') or 5
    word_recommendations = recommend_words(input_words, n_outputs=n_outputs, multiple=multiple)
    
    return jsonify(status=200,
                   input_words=input_words,
                   word_recs=word_recommendations)


if __name__ == '__main__':
  app.run()
