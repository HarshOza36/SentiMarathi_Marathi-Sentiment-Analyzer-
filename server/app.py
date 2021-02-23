from flask import Flask, render_template, url_for, request, session, redirect, flash
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

from google_trans_new import google_translator
tr = google_translator()

# from googletrans import Translator
# tr = Translator()

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# stop = stopwords.words('english-new')

print(os.listdir()) 
# so in heroku , it is running this file from the root but not from the server folder
# Hence all the paths will have to be from the root instead from the actual server paths
# So all the paths in this code are like that.

# Flask Config
# But, for Flask Paths below, it runs from inside the server folder hence it follows normal path conventions.
app = Flask(__name__, static_folder='../client/static',
            template_folder="../client/templates")

app.config["SECRET_KEY"] = "ursecretkey"

ENV = 'dev'

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False

# Custom stopwords

with open('server/helper/english-new', 'r') as f:
    stop = f.readlines()
    f.close()
stop = [i.rstrip() for i in stop]


@app.route('/')
def index():
    return render_template('index.html', res=3)


@app.route('/detect_sentiment', methods=['GET', 'POST'])
def detect_sentiment():
    inp = request.form["inp"]
    print(inp)
    # print(stop)
    # eng = tr.translate(inp).text
    eng = tr.translate(inp)
    eng = eng.lower().replace('\W+', " ").replace("'", " ")
    removed_stopword = []
    for word in eng.split():
        if word not in stop:
            removed_stopword.append(word)
    # eng = np.array([" ".join(removed_stopword)])
    eng = [" ".join(removed_stopword)]
    print(eng)
    tfidfconverter = TfidfVectorizer(max_features=200, min_df=1, max_df=0.10)
    tfidf_model = pickle.load(open('training/models/tfidf.pkl', 'rb'))
    x = tfidf_model.transform(eng).toarray()
    models = ['lr.pkl', 'dt.pkl', 'gnb.pkl', 'knn.pkl', 'rfc.pkl', 'svm.pkl']
    for i in models:
        model = pickle.load(open('training/models/'+i, 'rb'))
        print(i, " ", model.predict(x))
    model = pickle.load(open('training/models/rfc.pkl', 'rb'))
    pred = model.predict(x)
    print(pred)
    res = 0
    if(pred[0] == 1):
        res = 1
    elif(pred[0] == 0):
        res = 0
    elif(pred[0] == -1):
        res = -1
    return render_template('index.html', res=res)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
