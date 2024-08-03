from sqlalchemy import Column, Integer, String, DateTime, Text, UUID, Boolean, Float, ARRAY
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Retry(Base):
    __tablename__ = 'retry'

    id = Column(Integer, primary_key=True, nullable=False, unique=True)
    voice_name = Column(String(256), primary_key=False, nullable=False, unique=False)
    text = Column(String(2048), nullable=False, unique=False)
    file_list = Column(ARRAY(String), nullable=False, unique=False)
    # List of wav files to use for sampling
    bad_output_wav = Column(ARRAY(Float), nullable=False, unique=False)
    date_uploaded = Column(DateTime, nullable=False, unique=False)


    def __init__(self, voice_name, file_list, text, bad_output_wav):
        self.voice_name = voice_name
        self.file_list = file_list
        self.text = text
        self.bad_output_wav = bad_output_wav
        self.date_uploaded = datetime.now()


    def __str__(self):
        return f'DocumentEntry(text={self.text}, bad_output_wav= {self.bad_output_wav})'


    def to_dict(self):
        return {
            "id" : str(self.id),
            "text" : self.text,
            "voice_name" : self.voice_name,
            "file_list" : self.file_list,
            "bad_output_wav" : self.bad_output_wav,
            "date_uploaded" : self.date_uploaded
        }