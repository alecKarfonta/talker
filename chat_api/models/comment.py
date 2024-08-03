from sqlalchemy import Column, Integer, String, DateTime, Text, UUID, Boolean, Float

from datetime import datetime
#from DatabaseFactory import DatabaseFactory
from uuid import uuid4
# User imports
from shared.color import Color
from controllers.sentiment import SentimentScore

from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Comment(Base):
    __tablename__ = 'comment'

    id = Column(UUID, primary_key=True, unique=True, nullable=False)
    response_to_id = Column(UUID, nullable=True)
    time = Column(DateTime, nullable=False)
    commentor = Column(String, nullable=False)  
    comment = Column(Text, nullable=False)
    positive_sentiment = Column(Float, nullable=True)
    negative_sentiment = Column(Float, nullable=True)

    def __init__(self, 
                 commentor:str, 
                 comment:str,
                 sentiment:SentimentScore=None, 
                 positive_sentiment:float=None,
                 negative_sentiment:float=None,
                 useDatabase:bool=True,
                 response_to_id:UUID=None
                ):
        self.id = str(uuid4())#
        self.response_to_id = response_to_id
        self.time = datetime.now()
        self.commentor = commentor
        self.comment = comment
        #self.positive_sentiment = positive_sentiment
        #self.negative_sentiment = negative_sentiment
        self.sentiment = sentiment
        if sentiment:
            self.positive_sentiment = float(sentiment.positive_score)
            self.negative_sentiment = float(sentiment.negative_score)
        self._isDatabase = useDatabase
        #if (useDatabase):
        #    super().__init__(debug=True)
            

    def __repr__(self) -> str:
        return f"{self.commentor}: {self.comment}"
        
    def get_age(self):
        curr = datetime.now()
        delta = curr - self.time
        return delta
        
    def printf(self):
        if self.positive_sentiment > .75:
            return f"{Color.F_Green}{self.commentor}: {self.comment}{Color.F_White}"
        elif self.negative_sentiment > .75:
            return f"{Color.F_Red}{self.commentor}: {self.comment}{Color.F_White}"
        else:
            return f"{Color.F_Blue}{self.commentor}: {self.comment}{Color.F_White}"
    
    def prompt(self):
        return f"{self.commentor}: {self.comment}"

    def to_dict(self):
        return {
            "id" : self.id,
            "time" : self.time,
            "commentor" : self.commentor,
            "comment" : self.comment,
            "positive_sentiment" : self.positive_sentiment,
            "negative_sentiment" : self.negative_sentiment
        }
    
    def __str__(self):
        return str(self.to_dict())
        
        