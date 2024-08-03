from color import Color
from sentiment import SentimentScore
from datetime import datetime
#from DatabaseFactory import DatabaseFactory
from uuid import uuid4

class Comment():
    def __init__(self, commentor:str, comment:str, sentiment:SentimentScore,useDatabase:bool=True):
        self.id = str(uuid4())
        self.time = datetime.now()
        self.commentor = commentor
        self.comment = comment
        self.sentiment = sentiment
        self._isDatabase = useDatabase
        if (useDatabase):
            super().__init__(debug=True)
            

    def __repr__(self) -> str:
        return f"{self.commentor}: {self.comment}"
        
    def get_age(self):
        curr = datetime.now()
        delta = curr - self.time
        return delta
        
    def printf(self):
        if self.sentiment.sentiment == "positive":
            return f"{Color.F_Green}{self.commentor}: {self.comment}{Color.F_White}"
        if self.sentiment.sentiment == "neutral":
            return f"{Color.F_Blue}{self.commentor}: {self.comment}{Color.F_White}"
        if self.sentiment.sentiment == "negative":
            return f"{Color.F_Red}{self.commentor}: {self.comment}{Color.F_White}"
    
    def prompt(self):
        return f"{self.commentor}: {self.comment}"
        
        