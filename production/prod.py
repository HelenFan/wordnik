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

def get_clean_data(path):
    df = pd.read_json(path)
    for c in df.columns:
        df[c] = df[c].apply(lambda x: list(x.values())[0] if type(x) == dict else x)
    return df

def get_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data

data_path = Path('/home/gcmac/prod/prod_data')

@app.before_first_request
def load_data():
    global listed_words
    global list_popularity
    global search_popularity
    global tag_popularity
    listed_words = get_clean_data(data_path/'valid_listed_words')
    list_popularity = dict(listed_words.lcword.value_counts())
    search_popularity = get_json(data_path/'word_cnts.json')
    tag_popularity = get_json(data_path/'word_numtags_map.json')

    global model
    model_path = data_path/'wiki.en.vec'
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)


def find_rank(lst, dic):
    print("in find_rank")
    words_in_pop_list = [w for w in lst if w in dic]
    words_not_in_pop_list = [w for w in lst if w not in dic]
    df_in_list = pd.DataFrame(
        [dic[w] for w in words_in_pop_list],
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


def filter_words(input_words, lst, stemmer):
    input_stems = set([stemmer.stem(i) for i in input_words])
    return [w for w in lst if stemmer.stem(w) not in input_stems]


def find_words(input_words, n_outputs=1):
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

    # get list of rank of candidate words by popularity of list, search, tag on wordnik
    rank_list = find_rank(filtered_candidates, dic=list_popularity)
    rank_search = find_rank(filtered_candidates, dic=search_popularity)
    rank_tag = find_rank(filtered_candidates, dic=tag_popularity)

    # concat vectors from above and join
    df = pd.concat([rank_list, rank_search, rank_tag], axis=1, sort=True).reset_index()
    df['rank_sum'] = df['rank'].sum(axis=1)
    df['stem'] = df['index'].apply(lambda x: stemmer.stem(x))
    df.drop_duplicates(['stem'],inplace=True)
    return list(df.sort_values('rank_sum')['index'].values[:n_outputs])


@app.route('/')
def hello_world():
    import numpy as np
    return 'Hello from Flask!'

@app.route('/word_pred/', methods=['POST'])
def word_pred():
    data = request.get_json()
    input_words = data['input_words']
    word_recommendations = recommend_words(input_words, n_outputs=10, multiple=5)
    
    return jsonify(status=200,
                   input_words=input_words,
    #               word_recs=['this','was','a','test'])
                   word_recs=word_recommendations)


if __name__ == '__main__':
  app.run()
