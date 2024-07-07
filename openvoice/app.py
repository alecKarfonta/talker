import json
import os,sys
import logging
from flask import Flask, jsonify, request, render_template
from pydantic import BaseModel
from flask_swagger_ui import get_swaggerui_blueprint

import numpy as np
import soundfile as sf
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
        'app_name': 'TTS API'
    }
)

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
speed = 1.0

# Init tone converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# Init Speakers
reference_speaker = 'resources/major/major_2_02.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

#logger.debug(f"{__name__}(): Loading default speaker: {reference_speaker = }")
#logger.debug(f"{__name__}(): {type(target_se) = }")
#logger.debug(f"{__name__}(): {len(target_se) = }")
#logger.debug(f"{__name__}(): {target_se[0:10] = }")


speaker_key = "es"
speaker_path = f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth'
logger.debug(f"{__name__}(): Loading source speaker: {speaker_path = }")
source_se = torch.load(speaker_path, map_location=device)
logger.debug(f"{__name__}(): {source_se[0][0:10] = }")

voices = {}

voices["Major"] = target_se


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


def run_tts(input_text:str, model, output_dir, source_se, target_se, speaker_id=0, speed=1.0, tau=0.3):
    
    logger.debug(f"run_tts({input_text = }, model, {output_dir = }, ), ")

    src_path = f'{output_dir}/tmp.wav'
    
    logger.debug(f"run_tts(): Sending text to base model")
    model.tts_to_file(input_text, speaker_id, src_path, speed=speed)
    logger.debug(f"run_tts(): Saved base audio to {src_path}")

    logger.debug(f"run_tts(): Convert to speaker")
    #print(f"\t Converting to speaker: {target_se["audio_name"]}")
    save_path = f'{output_dir}/output_v2_{input_text[0:10]}.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        tau=tau,
        message=encode_message)
    logger.debug(f"run_tts(): File saved to {save_path}")

    # Read the wav file back in
    samplerate, wav = wavfile.read(save_path)

    samplerate, base_wav = wavfile.read(src_path)

    return wav, samplerate, base_wav


@app.get("/test")
def root():
    return {"message": "All good"}, 200


@app.post("/tts")
def tts():
    data = request.json
    text = data.get('text', 'Hello World')
    voice = data.get('voice_name', 'Major')
    tau = data.get('tau', 0.3)
    target_se = voices[voice]

    #logger.debug(f"tts(): {type(target_se) = }")
    #logger.debug(f"tts(): {len(target_se) = }")
    #logger.debug(f"tts(): {target_se[0:10] = }")
    #logger.debug(f"tts(): {target_se[0] = }")

    logger.info(f'tts(): Reading "{text = }" with {voice = }')

    logger.debug(f"{__name__}(): {source_se[0][0:10] = }")
        

    wav, samplerate, base_wav = run_tts(text, model, output_dir, source_se, target_se, speaker_id=0, speed=1.0, tau=tau)
    return jsonify({"wav": wav.tolist(), "base_wav": base_wav.tolist(), "samplerate": samplerate})


@app.route('/list_voices', methods=['GET'])
def list_voices():
    return jsonify({"voices": list(voices.keys())}), 200


@app.route('/base_speakers', methods=['GET'])
def base_speakers():
    speaker_ids = model.hps.data.spk2id

    return jsonify({"speakers_keys": list(speaker_ids.keys())}), 200


@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    data = request.get_json()
    wav_float_array = np.array(data['wav'], dtype=np.float32)
    voice_name = data['voice_name']
    
    # Save the wav float array to a file
    folder_path = f'resources/data/'
    # Make the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    file_path = f'{folder_path}/{voice_name}.wav'

    sf.write(file_path, wav_float_array, 22050)
    
    # Load the file into target_se
    target_se = se_extractor.get_se(file_path, tone_color_converter, vad=True)
    #logger.debug(f"upload_voice(): {type(target_se) = }")
    #logger.debug(f"upload_voice(): {len(target_se) = }")
    #logger.debug(f"upload_voice(): {target_se[0][0:10] = }")
    #logger.debug(f"upload_voice(): {target_se[0] = }")
    
    # Store the target_se in the dictionary
    voices[voice_name] = target_se[0]
    
    return jsonify({"message": "File uploaded and processed successfully", "file_name": voice_name}), 200


@app.route('/update_source_se', methods=['POST'])
def update_source_se():
    data = request.get_json()
    key = data['speaker_key']
    checkpoint_path = f'checkpoints_v2/base_speakers/ses/{key}.pth'
    
    logger.debug(f"{__name__}(): {checkpoint_path = }")
    
    if os.path.isfile(checkpoint_path):
        source_se = torch.load(checkpoint_path, map_location=device)

        logger.debug(f"{__name__}(): {source_se[0][0:10] = }")

        logger.debug(f"{__name__}(): Updated source_se with key {key}")
        return jsonify({"message": "source_se updated successfully", "key": key}), 200
    else:
        logger.debug(f"{__name__}(): Key {key} not found")
        return jsonify({"message": "Key not found", "key": key}), 404


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)