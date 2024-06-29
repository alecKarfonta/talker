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

# User imports
from controllers.conversation import Conversation
from controllers.robot import Robot
from models.comment import Comment
from models.human import Human

# Pre-load nltk libs
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("espeakng").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('chat_api.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

SERVICE_PORT = os.environ.get('SERVICE_PORT', 5001)
SERVICE_HOST = os.environ.get('SERVICE_HOST', "0.0.0.0")
SERVICE_DEBUG = os.environ.get('SERVICE_DEBUG', 'True')

DATABASE_TYPE = os.environ.get('DATABASE_TYPE', "postgresql")
DATABASE_USERNAME = os.environ.get('DATABASE_USER', "test")
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD', "test")
DATABASE_SCHEMA = os.environ.get('DATABASE_SCHEMA', "chat")
DATABASE_HOST = os.environ.get('DATABASE_HOST', "127.0.0.1")
#TTS_URL = os.environ.get('SWC_TTS_URL', "http://localhost:8100/tts")

SWAGGER_URL="/swagger"
API_URL="/static/swagger.yaml"


swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Chat API'
    }
)


app = Flask(__name__,static_folder="cache")
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
SessionClass = sessionmaker(bind=engine)

# Init robot
robot = Robot(
              name="Major",
              persona="A female military cyborg police officer living in the year 2032.",
              model_name="l3utterfly/mistral-7b-v0.1-layla-v4",
              is_use_bnb=True,
              is_use_gpu=True,
             )

# Init coversation
conversation = Conversation(robot=robot)

def init_tables():
    # Init tables
    try:
        Comment.__table__.create(engine)
    except sqlalchemy.exc.ProgrammingError:
        pass


@app.get("/test")
async def root():
    return {"message": "All good"}, 200



@app.route('/comment', methods=['POST']) 
async def comment():
    logger.debug(f"comment()")
    #logger.debug(f"comment(): {message = }")
    #logger.debug(f"comment(): {api_comment.data = }")

    data = request.get_json()
    logger.debug(f"comment(): {data = }")

    user_comment = data.get("comment", None)
    user_commentor = data.get("commentor", None)

    logger.debug(f"comment(): {user_comment = }")
    logger.debug(f"comment(): {user_commentor = }")

    
    response, wav = conversation.process_comment(commentor=user_commentor, comment=user_comment, is_speak_response=True)

    return {"message": "All good", "response": response, "wav" : json.dumps(wav)}, 200




# Run the Flask app
if __name__ == '__main__':
    #init_tables()
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)