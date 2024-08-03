from sqlalchemy import Column, Integer, String, DateTime, Text, UUID, Boolean, Float, ARRAY
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class GeneratedAudio(Base):
    __tablename__ = 'generated_audio'

    id = Column(Integer, primary_key=True, nullable=False, unique=True)
    model_name = Column(String(256), primary_key=False, nullable=False, unique=False)
    voice_name = Column(String(256), primary_key=False, nullable=False, unique=False)
    #file_list = Column(ARRAY(String), nullable=False, unique=False)
    text = Column(String(4096), nullable=False, unique=False)
    # List of wav files to use for sampling
    wav = Column(ARRAY(Float), nullable=False, unique=False)
    is_bad = Column(Boolean, nullable=False, unique=False)
    date_uploaded = Column(DateTime, nullable=False, unique=False)


    def __init__(self, model_name, voice_name, text, wav, is_bad=False):
        self.model_name = model_name
        self.voice_name = voice_name
        #self.file_list = file_list
        self.text = text
        self.wav = wav
        self.is_bad = is_bad
        self.date_uploaded = datetime.now()


    def __str__(self):
        return f'DocumentEntry(text={self.text}, wav= {self.wav})'


    def to_dict(self):
        return {
            "id" : str(self.id),
            "model_name" : self.model_name,
            "voice_name" : self.voice_name,
            #"file_list" : self.file_list,
            "text" : self.text,
            "wav" : self.wav,
            "is_bad" : self.is_bad,
            "date_uploaded" : self.date_uploaded
        }