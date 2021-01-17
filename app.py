import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
import pickle
import praw
import pandas as pd
import re
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize , sent_tokenize
from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

ML_model = pickle.load(open('model.pkl', 'rb'))
NLP_model = Doc2Vec.load("doc2vec.model")
reddit = praw.Reddit(client_id='isECSn1zxjro_w',
                         client_secret='sIH18LHw6mCHAxZu5BvIHh5ndkDpTQ',
                         user_agent='Reddit')

def get_data(url_):
    data = {}

    comment = reddit.comment(url = url_)
    comment_id = comment.id
    body = comment.body
    if comment.is_root == False:
        parent_id = comment.parent_id
        parent_score = int(score(parent_id))
        time_since_parent = time_since_comment(parent_id, comment.created_utc)
        comment_tree_root_id = comment_tree_root(comment.id)
        if comment_tree_root_id == False:
            comment_tree_root_score = 0
            time_since_comment_root = 0
        else:
            comment_tree_root_score = score(comment_tree_root_id)
            time_since_comment_root = time_since_comment(comment_tree_root_id , comment.created_utc)
        parent_cosine , parent_euc = get_angle(parent_id , body)
    else:
        parent_score = 0
        time_since_parent = 0
        comment_tree_root_score = 0
        time_since_comment_root = 0
        parent_cosine = 0
        parent_euc = 0
    x = [parent_score, time_since_parent, comment_tree_root_score,
       time_since_comment_root, parent_cosine, parent_euc]
    return x
    
def score(id_):
    x = str(id_)
    if (x[:3] == 't1_' or x[:3] == 't3_'):
        x = id_[3:]
        try:
            parent = reddit.comment(x)
            parent_score = parent.score
            return int(parent_score)
        except:
            parent = reddit.submission(x)
            parent_score = parent.score
            return int(parent_score)
    elif x[:6] == 't1_t3_':
        x = id_[6:]
        try:
            parent = reddit.comment(x)
            parent_score = parent.score
            return int(parent_score)
        except :
            comment = reddit.submission(x)
            parent_score = parent.score
            return int(parent_score)
    else:
        x = id_
        try:
            parent = reddit.comment(x)
            parent_score = parent.score
            return int(parent_score)
        except:
            parent = reddit.submission(x)
            parent_score = parent.score
            return int(parent_score)

def comment_tree_root(x):
    if (x[:3] == 't1_' or x[:3] == 't3_'):
        x = x[3:]
    comment = reddit.comment(x)
    if (comment.is_root == True):
        return x
    else:
        try:
            return comment_tree_root(comment.parent_id)
        except:
            return False
            
    
def time_since_comment(id_, comment_time):
    x = str(id_)
    if x[:6] == 't1_t3_':
        x = id_[6:]
        try:
            root = reddit.comment(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()

        except:
            root = reddit.submission(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()
    elif (x[:3] == 't3_' or x[:3] == 't1_'):
        x = id_[3:]
        try:
            root = reddit.comment(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()

        except:
            root = reddit.submission(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()
    else:
        x = id_
        try:
            root = reddit.comment(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()

        except:
            root = reddit.submission(x)
            root_time = root.created_utc
            return (pd.to_datetime(root_time) - pd.to_datetime(comment_time)).total_seconds()

def preprocess(text):
    text = [text.lower()] 
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text)) 
    text = text.split()
    words = [w for w in text if w not in stopwords.words("english")]
    return words              
                
def get_angle(id_ , body):
    if (id_[:3] == 't1_' or id_[:3] == 't3_'):
        x = id_[3:]
        try:
            parent = reddit.comment(x)
            parent_body = parent.body
        except:
            parent = reddit.submission(x)
            parent_body = parent.title
    elif id_[:6] == 't1_t3_':
        x = id_[6:]
        try:
            parent = reddit.comment(x)
            parent_body = parent.body
        except:
            comment = reddit.submission(x)
            parent_body = parent.title
    comment_body = preprocess(body)
    parent_body = preprocess(parent_body)
    child = NLP_model.infer_vector(comment_body)
    parent = NLP_model.infer_vector(parent_body)
    cos = cosine(child , parent) 
    euc = euclidean(child , parent)
    return cos,euc
    
@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict' , methods = ['POST'])

def predict():
    url = str(flask.request.form['url'])
    X_test = get_data(url)
    
    result = int(ML_model.predict([X_test]))
    return render_template('Index.html', results = 'Predicted score of the comment is: {}'.format(result))

if __name__ == "__main__":
    app.run(debug = True)
    
    
    