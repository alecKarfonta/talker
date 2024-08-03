from sqlalchemy import Column, Integer, String, DateTime, Text, UUID, Boolean, Float, ARRAY
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Voice(Base):
    __tablename__ = 'voice'

    id = Column(UUID, primary_key=True, nullable=False, unique=True)
    voice_name = Column(String(100), nullable=False, unique=False)
    # List of wav files to use for sampling
    file_list = Column(ARRAY(String), nullable=False, unique=False)
    # A fallback list of files to pull from when generation fails
    fallback_file_list = Column(ARRAY(String), nullable=False, unique=False)
    # Percent speed up to apply to output audio
    default_speed = Column(Float, nullable=False, unique=False)
    # Expected read speed value to use when evaluating for bad output
    expected_read_speed = Column(Float, nullable=False, unique=False)
    date_uploaded = Column(DateTime, nullable=False, unique=False)


    def __init__(self, voice_name, file_list, fallback_file_list, default_speed, expected_read_speed=2.0):
        self.voice_name = voice_name
        self.file_list = file_list
        self.fallback_file_list = fallback_file_list
        self.default_speed = default_speed
        self.expected_read_speed = expected_read_speed
        self.date_uploaded = datetime.now()


    def __str__(self):
        return f'DocumentEntry(voice_name={self.voice_name}, file_list= {self.file_list})'


    def to_dict(self):
        return {
            "id" : str(self.id),
            "voice_name" : self.voice_name,
            "file_list" : self.file_list,
            "date_uploaded" : str(self.date_uploaded)
        }