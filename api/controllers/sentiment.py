import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import csv
import urllib.request

class SentimentScore():
    def __init__(self, sentiment:str, positive_score:float, neutral_score:float, negative_score:float):
        self.sentiment = sentiment
        self.positive_score = positive_score
        self.neutral_score = neutral_score
        self.negative_score = negative_score
        
    def __repr__(self):
        return (f"sentiment = {self.sentiment} ({self.positive_score, self.neutral_score, self.negative_score})")
    

class Sentiment:
    def __init__(self, model_name:str = f"cardiffnlp/twitter-roberta-base-sentiment"):      
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ['negative', 'neutral', 'positive']
        
        #labels=[]
        #mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
        #with urllib.request.urlopen(mapping_link) as f:
        #    html = f.read().decode('utf-8').split("\n")
        #    csvreader = csv.reader(html, delimiter='\t')
        #self.labels = [row[1] for row in csvreader if len(row) > 1]

    def get_sentiment(self, text:str):
        """
        Given a text string, return the sentiment of the text.
        """
        try:
            #text = preprocess(text)
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]

            label_scores = {}
            for i in range(scores.shape[0]):
                l = self.labels[ranking[i]]
                s = scores[ranking[i]]
                label_scores[l] = s
            #return [labels[ranking[0]], label_scores["positive"], label_scores["neutral"], label_scores["negative"]]
            return {"sentiment" : self.labels[ranking[0]],
                    "positive_score" : label_scores["positive"], 
                    "neutral_score" : label_scores["neutral"], 
                    "negative_score" : label_scores["negative"]
                   }
        except:
            print (f"Sentiment(): Failed to get sentiment for text = {text}")
            return {"sentiment" : "error",
                    "positive_score" : 0.0, 
                    "neutral_score" : 0.0, 
                    "negative_score" : 0.0
                   }