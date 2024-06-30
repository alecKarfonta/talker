import json
import os,sys
import logging
from flask import Flask, jsonify, request, render_template
from pydantic import BaseModel
from flask_swagger_ui import get_swaggerui_blueprint


import os
import glob
import torch
from melo.api import TTS
from scipy.io import wavfile



from openvoice import se_extractor
from openvoice.api import ToneColorConverter

import sys
#sys.path.append("..") 

# User imports


# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('tts_api.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

SERVICE_PORT = os.environ.get('SERVICE_PORT', 5000)
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
        'app_name': 'TTS API'
    }
)

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
output_dir = 'outputs_v2'
speed = 1.0

# Init tone converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# Init Speakers
reference_speaker = 'resources/major/major_2_02.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

speaker_key = "en-newest"
source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)

#  Start the Flask app
app = Flask(__name__, static_folder="cache")
app.debug = True
#db_uri = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
#app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
#logging.info(f"Connecting to db: {db_uri}")
#db = SQLAlchemy(app)
#engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
#SessionClass = sessionmaker(bind=engine)

#def init_tables():
#    # Init tables
#    try:
#        Comment.__table__.create(engine)
#    except sqlalchemy.exc.ProgrammingError:
#        pass


def init_model(device:str="cpu", language = "EN_NEWEST"):
    model = TTS(language=language, device=device)
    return model

model = init_model(device=device)


def run_tts(input_text:str, model, output_dir, source_se, target_se, speaker_id=0, speed=1.0):
    
    src_path = f'{output_dir}/tmp.wav'
    
    print(f"Reading: {input_text}")
    model.tts_to_file(input_text, speaker_id, src_path, speed=speed)

    #print(f"\t Converting to speaker: {target_se["audio_name"]}")
    save_path = f'{output_dir}/output_v2_{input_text[0:10]}.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)

    # Read the wav file back in
    samplerate, wav = wavfile.read(save_path)

    return wav, samplerate


@app.get("/test")
async def root():
    return {"message": "All good"}, 200


@app.post("/tts")
def tts():
    data = request.json
    text = data.get('text', 'Hello World')
    logger.info(f"Received text: {text}")
    wav, samplerate = run_tts(text, model, output_dir, source_se, target_se, speaker_id=0, speed=1.0)
    return jsonify({"wav": wav.tolist(), "samplerate": samplerate})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)