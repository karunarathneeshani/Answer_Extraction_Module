import nltk
nltk.download('stopwords')
import tweepy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english')) - set(['not'])
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string
import re
import pymongo
from collections import Counter
from owlready2 import *
from rdflib.plugins.sparql import prepareQuery
from rdflib import URIRef

ACCESS_TOKEN = "4834000149-5pxyPG3rko5NrG3zsu0l7LZcH2aBKQYUgWvmn98"
ACCESS_TOKEN_SECRET = "e4Ecy03foBR7Cx7F0Wp6w8GUYX2vAsgOZkD0PcKQktLwu"
CONSUMER_KEY = "TWh3Q5c36GCoKfwOqmPdrmxOs"
CONSUMER_SECRET = "edWP05D3UnYJutg68L6jBu9THY6oUJn4MmugsYrpgk4eHmdQRc"

connection = pymongo.MongoClient('localhost', 27017)
database = connection['category_base']
collection = database ['uber_category']


# import twitter_credentials


class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamer():
    """
        Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, user):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        # api = tweepy.API(auth)
        stream = tweepy.Stream(auth,listener)
        stream.filter(track=user)


class TwitterListener(tweepy.StreamListener):
    """
    This is a basic listener class that just prints received tweets to stdout.
    """

    def __init__(self,fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        tweettext = bagOfWords()
        try:
            with open(self.fetched_tweets_filename, 'a') as tf:
                json_load = json.loads(data)
                text = json_load['text']
                tweettext.preprocess_text(text)
                #tf.write(text) can write in a text file for further store
                #tf.write('\n')
            return text
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_status(self, status):
        print(status.text)

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status.text)


# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print tweet.text

class preProcessing():

    def tweettxt(self,text):
        print(text)
        return True

    def remove_pattern(self,input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    def remove_punct(self,text):
        text=text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def removeStopWords(self,text):
        clean_word_list = [word for word in text.split() if word not in stoplist]
        return clean_word_list

    def deEmojify(self,inputString):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', inputString)

    def tokenizer(self,text):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(text).todense()


class bagOfWords():

    def preprocess_text(self, tweet):
        preprocess=preProcessing()
        removeURL = preprocess.remove_pattern(tweet,"http\S+")
        removeMentions = preprocess.remove_pattern(removeURL,"@[\w]*")
        removePuncs = preprocess.remove_punct(removeMentions)
        removeEmoji = preprocess.deEmojify(removePuncs)
        lemmatizer = WordNetLemmatizer()
        lemmatizedwords= lemmatizer.lemmatize(removeEmoji)
        a = preprocess.removeStopWords(lemmatizedwords)
        prop = collection.find({})
        for documents in prop:
            b = documents['Key words']
            a_vals = Counter(a)
            b_vals = Counter(b)

            # convert to word-vectors
            words = list(a_vals.keys() | b_vals.keys())
            a_vect = [a_vals.get(word, 0) for word in words]  # [0, 0, 1, 1, 2, 1]
            b_vect = [b_vals.get(word, 0) for word in words]  # [1, 1, 1, 0, 1, 0]

            # find cosine
            len_a = sum(av * av for av in a_vect) ** 0.5  # sqrt(7)
            len_b = sum(bv * bv for bv in b_vect) ** 0.5  # sqrt(4)
            dot = sum(av * bv for av, bv in zip(a_vect, b_vect))  # 3
            cosine = dot / (len_a * len_b)

            if (cosine > 0.6):
                print(documents['Property'])
                onto = get_ontology("file://E:/Academic/Final.owl").load()
                graph = default_world.as_rdflib_graph()
                c = documents['Property']
                UC = URIRef('http://www.semanticweb.org/hp/ontologies/2019/8/FinalProject#')
                q = prepareQuery('''SELECT ?o
                                      WHERE {
                                                ?subject UC:''' + c + ''' ?object;
                UC:answer ?o.}''', initNs={'UC': UC})

                results = graph.query(q)
                response = []
                for item in results:
                    o = str(item['o'].toPython())
                    o = re.sub(r'.*#', "", o)
                    response.append(o)
                    print(response)

        return True


if __name__ == '__main__':
    user = ["@gethma_perera"]
    fetched_tweets_filename = "tweets.txt"

    twitter_streamer = TwitterStreamer()
    x = twitter_streamer.stream_tweets(fetched_tweets_filename,user)
    print(x)


