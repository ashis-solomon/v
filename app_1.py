from flask import Flask, render_template, request
import json
import pandas as pd
import pickle
import joblib
import snscrape.modules.twitter as sntwitter
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio',methods=['POST','GET'])
def audio():
    displayName = request.args.get("displayName")
    print(displayName)
    trkID = request.args.get("trkID")
    print(trkID)
    trkPop = request.args.get("trkPop")
    print(trkPop)
    return render_template('audio.html',displayName=displayName,trkID=trkID,trkPop=trkPop)


@app.route('/stat',methods=['POST','GET'])
def stat():
    pass_str = request.form.get('hoo')
    items=pass_str.split('!@#$%')
    #print(items)
    print("hey")
    x=json.loads(items[0])
    y=pd.DataFrame(x)
    df=y.sort_values(by=['popularity'], ascending=False)
    df.drop(['type','uri','track_href','id','analysis_url'], axis=1,inplace=True)
    dfx=df.drop_duplicates(subset=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms', 'time_signature', 'name', 'explicit'] ,keep='first',ignore_index=False)
    dfxx=dfx.drop(['key','mode','explicit','time_signature','name'], axis=1)
    dfxx = dfxx.add_suffix('_ar')
    df_ar = dfxx[['danceability_ar','energy_ar','loudness_ar','speechiness_ar','duration_ms_ar','acousticness_ar','instrumentalness_ar','liveness_ar','valence_ar','tempo_ar','popularity_ar']].mean()
    dfdict=df_ar.to_dict()
    print(dfdict)
    print("hey")
    yy=json.loads(items[2])
    yy["year"]=float(yy["release_date"][:4])
    yy.pop("release_date")
    #if yy["mode"]==0:
       # yy["minor"]=1.0
       # yy["major"]=0.0
   # else:
        #yy["minor"]=0.0
        #yy["major"]=1.0
    #yy.pop('mode')
    yy.pop('s_id')
    yy.pop('time_signature')
    if yy["explicit"] == False:
        yy["explicit"]=0.0
    else:
        yy["explicit"]=1.0
    for i in range(12):
        yy[f'key_{i}']=0.0
    yy[f'key_{yy["key"]}']=1.0 
    yy.pop("key")
    yy.pop("key_0")
    print(yy)
    print("hey")

    acousticness = float(yy["acousticness"])
    danceability = float(yy["danceability"])
    duration_ms = float(yy["duration"])
    energy = float(yy["energy"])
    explicit = float(yy["explicit"])
    instrumentalness = float(yy["instrumentalness"])
    liveness = float(yy["liveness"])
    loudness = float(yy["loudness"])
    speechiness = float(yy["speechiness"])
    tempo = float(yy["tempo"])
    valence = float(yy["valence"])
    year = float(yy["year"])
    #key_0 = yy["key_0"]
    key_1 = yy["key_1"]
    key_2 = yy["key_2"]
    key_3 = yy["key_3"]
    key_4 = yy["key_4"]
    key_5 = yy["key_5"]
    key_6 = yy["key_6"]
    key_7 = yy["key_7"]
    key_8 = yy["key_8"]
    key_9 = yy["key_9"]
    key_10 = yy["key_10"]
    key_11 = yy["key_11"]
    mode = float(yy["mode"])
    #major = yy["major"]
    acousticness_ar = float(dfdict["acousticness_ar"])
    danceability_ar = float(dfdict["danceability_ar"])
    duration_ms_ar = float(dfdict["duration_ms_ar"])
    energy_ar = float(dfdict["energy_ar"])
    instrumentalness_ar = float(dfdict["instrumentalness_ar"])
    liveness_ar = float(dfdict["liveness_ar"])
    loudness_ar = float(dfdict["loudness_ar"])
    speechiness_ar = float(dfdict["speechiness_ar"])
    tempo_ar = float(dfdict["tempo_ar"])
    valence_ar = float(dfdict["valence_ar"])
    popularity_ar = float(dfdict["popularity_ar"])
    #popularity_ar = float(items[1])
    #if(popularity_ar<36):
    #   popularity_ar=float(items[1])
    model = joblib.load('model.pkl')
    prediction=model.predict([[acousticness,danceability,duration_ms,energy,explicit,instrumentalness,liveness,loudness,speechiness,tempo,valence,year,
    key_1,key_2,key_3,key_4,key_5,key_6,key_7,key_8,key_9,key_10,key_11,mode,acousticness_ar,
    danceability_ar,duration_ms_ar,energy_ar,instrumentalness_ar,liveness_ar,loudness_ar,speechiness_ar,tempo_ar,valence_ar,popularity_ar]])
    #prediction1=model.predict([[0.603,0.57,232013.0,0.212,0.0,0.0,0.148,-13.877,0.0437,122.513,0.182,1992.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.308159827586207,0.581103448275862,317261.637931034,0.564120689655173,0.092976407586207,0.228284482758621,-11.1943965517241,0.055008620689655,114.127551724138,0.454598275862069,28.9827586206896]])
    #print(prediction1)
    output=prediction[0]
    print(prediction)
    print(output)
    #print("hey")
   # print(items[2])
    # print(request.form)
# ----------------------------------------------------------------------------------------------------------------

    datearr = items[4].split("-")
    datearr[0] = int(datearr[0]) + 3
    
    tweets_list = []

    query = items[3].replace("-"," ")
    start_date = items[4]
    end_date = f'{datearr[0]}-{datearr[1]}-{datearr[2]}'

    print(query)
    print(start_date)
    print(end_date)

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} since:{start_date} until:{end_date}').get_items()):
        if i>100:
            break
        if(tweet.lang == 'en' and tweet.likeCount>0):
            tweets_list.append([tweet.content, tweet.url, tweet.date, tweet.likeCount, tweet.lang])
        
    cols = ['text', 'url', 'datetime', 'likeCount', 'lang']
    tweets_df = pd.DataFrame(tweets_list, columns=cols)
    tweets_df.sort_values(by=['likeCount'], inplace=True, ascending=False)
    tweets_df.reset_index(drop=True, inplace=True)

    print(len(tweets_df))
    print(tweets_df)
    
    tweets=[]
    for i in tweets_df['text']:
        tweets.append(i)

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = nltk.corpus.stopwords.words('english')

    def preprocess(textdata):
        processedText = []
        
        # Create Lemmatizer and Stemmer.
        wordLemm = WordNetLemmatizer()
        
        # Defining regex patterns.
        urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern       = '@[^\s]+'
        alphaPattern      = "[^a-zA-Z0-9]"
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        
        for tweet in textdata:
            tweet = tweet.lower()
            
            # Replace all URls with 'URL'
            tweet = re.sub(urlPattern,' URL',tweet)
            # Replace all emojis.
            for emoji in emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
            # Replace @USERNAME to 'USER'.
            tweet = re.sub(userPattern,' USER', tweet)        
            # Replace all non alphabets.
            tweet = re.sub(alphaPattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

            tweetwords = ''
            for word in tweet.split():
                # Checking if the word is a stopword.
                #if word not in stopwordlist:
                if len(word)>1:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')
                
            processedText.append(tweetwords)
            
        return processedText

    def load_models():
        # Load the vectoriser.
        file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
        vectoriser = pickle.load(file)
        file.close()
        # Load the LR Model.
        file = open('./Sentiment-LR.pickle', 'rb')
        LRmodel = pickle.load(file)
        file.close()
        
        return vectoriser, LRmodel

    def predict(vectoriser, model, text):
        # Predict the sentimentey
        textdata = vectoriser.transform(preprocess(text))
        sentiment = model.predict(textdata)
        
        # Make a list of text with sentiment.
        data = []
        for text, pred in zip(text, sentiment):
            data.append((text,pred))
            
        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(data, columns = ['tweet','sentiment'])
        df = df.replace([0,1], ["Negative","Positive"])
        return df

    vectoriser, LRmodel = load_models()
    df = predict(vectoriser, LRmodel, tweets)
    a=len(df[df["sentiment"]=="Positive"])
    b=len(df)
    print(a/b*100)
    repIndex = a/b*100
    repIndex = round(repIndex,2)
# ----------------------------------------------------------------------------------------------------------------
    return render_template('stat.html',pass_str=pass_str,output=output, repIndex=repIndex)

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/tryout')
def tryout():
    return render_template('tryout.html')

@app.route('/statTRY',methods=['GET'])
def statTRY():
    details = request.args.get("details")
    print(details)
    yy=json.loads(details)
    print(yy)
    if yy["explicit"] == "false" or yy["explicit"] == "False":
        yy["explicit"]=0.0
    else:
        yy["explicit"]=1.0
    for i in range(12):
        yy[f'key_{i}']=0.0
    yy[f'key_{yy["key"]}']=1.0 
    if yy["mode"] == 1:
        yy["mode"] = 1.0
    else:
        yy["mode"] = 0.0
    yy.pop("key")
    yy.pop("key_0")
    print("hey")
    print(yy)

    acousticness = float(yy["acousticness"])
    danceability = float(yy["danceability"])
    duration_ms = float(yy["duration"])
    energy = float(yy["energy"])
    explicit = float(yy["explicit"])
    instrumentalness = float(yy["instrumentalness"])
    liveness = float(yy["liveness"])
    loudness = float(yy["loudness"])
    speechiness = float(yy["speechiness"])
    tempo = float(yy["tempo"])
    valence = float(yy["valence"])
    #year = float(yy["year"])
    #key_0 = yy["key_0"]
    key_1 = yy["key_1"]
    key_2 = yy["key_2"]
    key_3 = yy["key_3"]
    key_4 = yy["key_4"]
    key_5 = yy["key_5"]
    key_6 = yy["key_6"]
    key_7 = yy["key_7"]
    key_8 = yy["key_8"]
    key_9 = yy["key_9"]
    key_10 = yy["key_10"]
    key_11 = yy["key_11"]
    mode = float(yy["mode"])

    model = joblib.load('statfinmodel.pkl')
    prediction=model.predict([[acousticness,danceability,duration_ms,energy,explicit,instrumentalness,liveness,loudness,speechiness,tempo,valence,
    key_1,key_2,key_3,key_4,key_5,key_6,key_7,key_8,key_9,key_10,key_11,mode]])
    #prediction1=model.predict([[0.603,0.57,232013.0,0.212,0.0,0.0,0.148,-13.877,0.0437,122.513,0.182,1992.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.308159827586207,0.581103448275862,317261.637931034,0.564120689655173,0.092976407586207,0.228284482758621,-11.1943965517241,0.055008620689655,114.127551724138,0.454598275862069,28.9827586206896]])
    #print(prediction1)
    output=prediction[0]
    print(output)



 #{'duration': 216100, 'loudness': -5.3, 'tempo': 63, 'acousticness': 0.08, 'danceability': 0.85, 'energy': 0.06, 'instrumentalness': 0.12,
 #'liveness': 0.5, 'speechiness': 0.25, 'valence': 0.54, 'mode': 1.0, 'explicit': 0.0, 'key_1': 1.0, 'key_2': 0.0, 'key_3': 0.0, 'key_4': 0.0, 'key_5': 0.0,
 #'key_6': 0.0, 'key_7': 0.0, 'key_8': 0.0, 'key_9': 0.0, 'key_10': 0.0, 'key_11': 0.0}



  # Index(['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
   #    'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness',
    #   'tempo', 'valence', 'year', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5',
     #  'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11', 'mode'],
      #dtype='object')







    #print(x)
    return render_template('statTRY.html',output=output)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "main":
    app.run(debug=True)
