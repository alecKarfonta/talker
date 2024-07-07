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
#sys.path.append("..") 

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


app = Flask(__name__, static_folder="cache")
app.debug = True
db_uri = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
logging.info(f"Connecting to db: {db_uri}")
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
SessionClass = sessionmaker(bind=engine)

characters = {
    "Major" : {
        "name" : "Major",
        "persona" : "A female military cyborg police officer living in the year 2032.",
    }
}

# Init robot
robot = Robot(
              name=characters["Major"]["name"],
              persona=characters["Major"]["persona"]
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


@app.get("/update_persona")
async def update_persona():
    character_name = request.args.get('character_name', None)
    new_persona = request.args.get('persona', None)
    if not new_persona and not character_name:
        logging.error(f"update_persona(): Must provide a character name or persona")
        return {"message": " Must provide a character name or persona"}, 400  
    logging.debug(f"update_persona(): {new_persona = }")

    if new_persona:
        robot.update_person(new_persona)
        return {"message": "All good"}, 200
    
    if character_name:
        if character_name not in characters:
            logging.error(f"update_persona(): Character name not found")
            return {"message": "Character name not found"}, 400
        robot.update_person(characters[character_name]["persona"])
        return {"message": "All good"}, 200


@app.route('/comment', methods=['POST']) 
async def comment_post():
    logger.debug(f"comment()")
    #logger.debug(f"comment(): {message = }")
    #logger.debug(f"comment(): {api_comment.data = }")

    data = request.get_json()
    logger.debug(f"comment(): {data = }")

    prompt = data.get('prompt', None)
    user_comment = data.get("comment", None)
    user_commentor = data.get("commentor", None)

    #logger.debug(f"comment(): {user_comment = }")
    logger.debug(f"comment(): {user_commentor = }")
    logger.debug(f"comment(): {type(user_commentor) = }")
    
    session = SessionClass()
    session.expire_on_commit = False
    user_comment, response_comment, wav = conversation.process_comment(commentor=user_commentor, comment=user_comment, prompt=prompt, is_speak_response=False)

    session.add(user_comment)
    session.add(response_comment)
    session.commit()
    session.close()

    logger.debug(f"comment(): {user_comment = }")
    logger.debug(f"comment(): {response_comment = }")

    logger.debug(f"comment(): {type(user_comment) = }")
    logger.debug(f"comment(): {type(response_comment) = }")
    # Write document entry to db

    return {"message": "All good", "response": response_comment.comment, "wav" : json.dumps(wav)}, 200


@app.route('/characters', methods=['GET'])
def get_characters():
    logger.debug(f"{__name__}(): Getting all characters")
    return jsonify({"characters": characters}), 200

@app.route('/characters', methods=['POST'])
def add_character():
    data = request.get_json()
    persona = data['persona']
    char_name = data['name']
    characters[char_name] = {
        "name" : char_name,
        "persona" : persona
    }
    
    logger.debug(f"{__name__}(): Added character with ID {char_id} and name {char_name}")
    
    return jsonify({"message": "Character added successfully", "id": char_id, "name": char_name}), 201

@app.route('/characters/<char_id>', methods=['GET'])
def get_character(char_id):
    logger.debug(f"{__name__}(): Getting character with ID {char_id}")
    
    character = characters.get(char_id)
    if character:
        return jsonify({"name": character["name"], "persona": character["persona"]}), 200
    else:
        return jsonify({"message": "Character not found"}), 404

@app.route('/characters/<char_id>', methods=['PUT'])
def update_character(char_id):
    data = request.get_json()
    char_name = data.get('name', None)

    if not char_name:
        return jsonify({"message": "Must provide a name to update"}), 400
    
    if char_id in characters:
        characters[char_id] = char_name
        logger.debug(f"{__name__}(): Updated character with ID {char_id} to new name {char_name}")
        return jsonify({"message": "Character updated successfully", "id": char_id, "name": char_name}), 200
    else:
        return jsonify({"message": "Character not found"}), 404

@app.route('/characters/<char_id>', methods=['DELETE'])
def delete_character(char_id):
    if char_id in characters:
        del characters[char_id]
        logger.debug(f"{__name__}(): Deleted character with ID {char_id}")
        return jsonify({"message": "Character deleted successfully", "id": char_id}), 200
    else:
        return jsonify({"message": "Character not found"}), 404



# Run the Flask app
if __name__ == '__main__':
    init_tables()
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)