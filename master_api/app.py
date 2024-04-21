import json
import os,sys
import logging
import requests
from uuid import uuid4 
from flask import Flask, jsonify, request, render_template
import sqlalchemy
from pydantic import BaseModel
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import select
from sqlalchemy.sql import text
from flask_swagger_ui import get_swaggerui_blueprint
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write

import sys
sys.path.append("..") 

# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('master_api.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

SERVICE_PORT = os.environ.get('SERVICE_PORT',"5000")
SERVICE_HOST = os.environ.get('SERVICE_HOST',"0.0.0.0")
SERVICE_DEBUG = os.environ.get('SERVICE_DEBUG','True')

CHAT_API_HOST = os.environ.get('CHAT_API_HOST',"192.168.1.196")
CHAT_API_PORT = os.environ.get('CHAT_API_PORT',"5001")

TTS_API_HOST = os.environ.get('TTS_API_HOST',"192.168.1.196")
TTS_API_PORT = os.environ.get('TTS_API_PORT',"8100")

DATABASE_TYPE = os.environ.get('DATABASE_TYPE',"postgresql")
DATABASE_USERNAME = os.environ.get('DATABASE_USER',"test")
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD',"test")
DATABASE_SCHEMA = os.environ.get('DATABASE_SCHEMA',"chat")
DATABASE_HOST = os.environ.get('DATABASE_HOST',"192.168.1.4")
TTS_URL=os.environ.get('SWC_TTS_URL',"http://localhost:8100/tts")

SWAGGER_URL="/swagger"
API_URL="/static/swagger.yaml"


swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Master API'
    }
)


app = Flask(__name__,static_folder="cache")
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
SessionClass = sessionmaker(bind=engine)


def init_tables():
    pass


@app.get("/test")
async def root():
    return {"message": "All good"}, 200



class Comment(BaseModel):
    comment: str 
    commentor: str


@app.post('/comment')
def comment():
    logger.debug(f"comment()")

    logger.debug(f"comment(): {request.data = }")

    content = request.json

    comment = content.get("comment", None)
    commenter = content.get("commenter", None)
    
    json = {
        "comment": comment,
        "commenter": commenter
    }

    logger.debug(f"{__name__}(): Making request to conversation api")
    # Make request to conversation api
    response = requests.post(url=f"http://{CHAT_API_HOST}:{CHAT_API_PORT}/comment", json=json)

    if response.status_code != 200:
        return {"message": "Error in conversation api"}, 500
    
    if "response" not in response.json():
        return {"message": "Error in conversation api"}, 500
    #if "wav" not in response.json():
    #    return {"message": "Error in conversation api"}, 500
    
    #wav = response.json()["wav"]
    user_response = message = response.json()["response"]

    logger.debug(f"{__name__}(): {user_response = }")

    logger.debug(f"{__name__}(): Making request to tts api")

    # Make request to tts api
    response = requests.post(url=f"http://{TTS_API_HOST}:{TTS_API_PORT}/tts", json={"text": user_response})

    if response.status_code != 200:
        return {"message": "Error in tts api"}, 500
    
    if "wav" not in response.json():
        return {"message": "Error in tts api"}, 500
    
    if "rate" not in response.json():
        return {"message": "Error in tts api"}, 500
    
    wav = response.json()["wav"]
    rate = response.json()["rate"]  

    logger.debug(f"{__name__}(): {np.mean(wav) = }")
    

    return {"message": "All good", "response": user_response}, 200





# Run the Flask app
if __name__ == '__main__':
    #init_tables()
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)