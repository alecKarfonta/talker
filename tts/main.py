"""
Microservice that accepts text inputs and stores them in a queue. Provides a method for pulling the queue. 
"""
import os
import time
from typing import Union
from types import NoneType
import uuid
import json
import redis
import logging
import zipfile
from zipfile import ZipFile
import threading
from copy import copy
import sqlalchemy
from nltk.tokenize import sent_tokenize
import numpy as np
from functools import reduce
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi import Request, Response
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import glob
# Init nltk
# TODO: Pre-init nltk in container
import nltk
nltk.download('punkt')

#from psycopg2.extensions import register_adapter, AsIs
#register_adapter(np.int64, AsIs)
#register_adapter(np.float32, AsIs)
#register_adapter(np.float16, AsIs)


# User imports
from voicebox import VoiceBox
from models.voice import Voice
from models.retry import Retry
from models.generated_audio import GeneratedAudio


# Environment Variables
DATABASE_TYPE = os.environ.get('DATABASE_TYPE',"postgresql")
DATABASE_USERNAME = os.environ.get('DATABASE_USER',"test")
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD',"test")
DATABASE_SCHEMA = os.environ.get('DATABASE_SCHEMA',"chat")
DATABASE_HOST = os.environ.get('DATABASE_HOST',"192.168.1.4")
# Init redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', "127.0.0.1")
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

port = os.environ.get("READER_PORT", 8100)

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

UPLOAD_FOLDER = "data"
DATA_FOLDER = "data"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# TODO: Base these values off of the expected read speed for the selected voice. It should be 
# based on some distance metric from this value and an expected variance. 
# Set a lower bound on read speed, when the words per second are below this level there is a problem.
read_speed_lower_threshold = 1.5
# Set an upper bound. Often when the model skips words the read_speed will be too high.
read_speed_upper_threshold = 5.0

# Connec to db 
db_uri = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
logger.debug(f"TTS(): Connecting to db {db_uri}")
engine = sqlalchemy.create_engine(db_uri)
SessionClass = sqlalchemy.orm.sessionmaker(bind=engine)


# Init tables
try:
    Voice.__table__.create(engine)
except sqlalchemy.exc.ProgrammingError:
    pass
try:
    Retry.__table__.create(engine)
except sqlalchemy.exc.ProgrammingError:
    pass
try:
    GeneratedAudio.__table__.create(engine)
except sqlalchemy.exc.ProgrammingError:
    pass


# Init voicebox
voice_box = VoiceBox(
            logger=logger, 
            config_filename="voicebox_config.json"
)

# Init api
app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

replace_list = [
    ("'", "'"),
    ("You're", "You are"),
    ("you're", "you are"),
    ("We're", "We are"),
    ("we're", "we are"),
    ("it's", "it is"),
    ("It's", "It is"),
    ("I'll", "I will"),
    ("i'll", "i will"),
    ("Major: nse", "Major:")
]
remove_list = ["#", "*", "</s>", 
               '"', 
               "@", "$", "%", 
               "^", "&", "*", 
               "(", ")", 
               #",", 
               "<unk>", 
               ":", ";", "\\n", 
               "nse:", "😁", "😉"]

class Message(BaseModel):
    command: str 
    text: str
    time: str
    priority: str
    command: str = None
    speed: float = None
    voice_clone: Union[str,list] = None


#import os
#for file in os.listdir("data/"):
#    if file.endswith(".wav"):

# TODO: Creat a more robust voice management system
voice_catalogue = {
    "major" : Voice(
                voice_name="major", 
                file_list=["data/major/major_11.wav","data/major/major_12.wav","data/major/major_13.wav","data/major/major_2_02.wav"],
                fallback_file_list=glob.glob("data/major/" +'/*.wav', recursive=False),
                default_speed=1.27,
                expected_read_speed=3.0
                ),
    "trump" : Voice(
                voice_name="trump", 
                file_list=["data/trump/trump_11.wav","data/trump/trump_12.wav","data/trump/trump_13.wav","data/trump/trump_14.wav","data/trump/trump_15.wav","data/trump/trump_16.wav","data/trump/trump_17.wav","data/trump/trump_18.wav","data/trump/trump_19.wav","data/trump/trump_21.wav","data/trump/trump_22.wav","data/trump/trump_23.wav","data/trump/trump_24.wav","data/trump/trump_25.wav","data/trump/trump_26.wav","data/trump/trump_27.wav","data/trump/trump_28.wav","data/trump/trump_29.wav","data/trump/trump_30.wav"],
                fallback_file_list=glob.glob("data/trump/" +'/*.wav', recursive=False),
                default_speed=1.011,
                expected_read_speed=2.0
                ),
    "dsp" : Voice(
                voice_name="dsp", 
                file_list=["data/dsp/dsp_07.wav","data/dsp/dsp_12.wav","data/dsp/dsp_13.wav","data/dsp/dsp_06.wav"],
                fallback_file_list=glob.glob("data/dsp/" +'/*.wav', recursive=False),
                default_speed=1.11,
                expected_read_speed=2.5
                ),
}

# Push voices in catalogue to db
#try:
session = SessionClass(expire_on_commit=False)
for voice_name in voice_catalogue:
    session.merge(voice_catalogue[voice_name])
session.commit()
#except:
#    logger.error(f"TTS(): Could not push voice to db")

session.close()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    #try:
    upload_filename = file.filename
    upload_name = upload_filename.replace(".zip", "")

    logger.debug(f"TTS.upload(): {upload_filename= }")
    # Save the uploaded file to the specified folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    logger.debug(f"TTS.upload(): Saving uploaded file to {file_path = }")
    try: 
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        f.close()
    except Exception as e:
        logger.debug(f"TTS.error(): Creating local file {upload_filename = }")

    extract_folder = os.path.join(UPLOAD_FOLDER, upload_name, "extracted")

    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    logger.debug(f"TTS.upload(): Extracting file at  {extract_folder = }")
    try:
        # Extract the contents of the ZIP file
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder) 

    except Exception as e:
        logger.debug( "what: " + e)
    # move all wav files to root


    for path in glob.glob(extract_folder +'/**/*.wav', recursive=True):
        os.rename(path,extract_folder +'/'+ os.path.basename(path))

    # Optional: Remove the uploaded ZIP file
    os.remove(file_path)

    # Identify all wav files in the upload

    # Loop over extracted files and find WAV files
    wav_files = []
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file.lower().endswith(".wav") and not file.startswith("."):
                wav_files.append(f"{extract_folder}/{file}")
    
    voice_catalogue[upload_name] = Voice(
                                    voice_name=upload_name,
                                    file_list=wav_files,
                                    fallback_file_list=wav_files,
                                    default_speed=1.08,
                                    expected_read_speed=2.0
    )
    # Push new voice to db
    session = SessionClass(expire_on_commit=False)
    session.merge(voice_catalogue[upload_name])
    session.commit()
    #except:
    #    logger.error(f"TTS(): Could not push voice to db")

    session.close()

    logger.debug(f"TTS.upload(): {voice_catalogue[upload_name].to_dict()}")

    return JSONResponse(content={"message": "File uploaded and extracted successfully", "extracted_folder": extract_folder})
    #except Exception as e:
    #    # Handle any exceptions and return an appropriate response
    #    return HTTPException(status_code=500, detail=str(e))
    

@app.get("/test/")
async def test():
    logger.debug(f"TTS.test()")
    return {"message": "All good"}


@app.get("/get_voice_list/")
async def get_voice_list():
    """
    Test return list of available voices.
    """
    logger.debug(f"TTS.get_voice_list()")
    return list(voice_catalogue.keys())


@app.get("/get_voice_details/")
async def get_voice_details(voice_name:str):
    logger.debug(f"TTS.get_voice_details()")
    if voice_name in voice_catalogue:
        return json.dumps(voice_catalogue[voice_name].to_dict())
    else:
        return "Unknown voice"


@app.post("/set-config/")
async def set_config(config: dict):
    """
    Given a config dict similar to voicebox_config.json, update the live voicebox instance
    with the new config values.
    """
    logger.debug(f"MessageQueue.set-config({config = })")
    voice_box.config = config
    voice_box.synth_params = config["synth_params"]
    voice_box.silence_filter_params = config["silence_filter_params"]
    #voice.speaker_wav = config["vocoder"]["speaker_wav"]
    voice_box.vocoder = config["vocoder"]
    #voice.__init__(config=config)
    return {"message": "Config updated"}


def push_to_queue(
            wav:np.ndarray, 
            rate:int = 24050, 
            text:str = "", 
            is_normalize:bool = True, 
            priority :str="1", 
            request_time :str=None 
    ):
    """
    Given a wav and accompanying metadata, push the wav data to the audio redis queue.

    Parameters:
    -----------
    wav : np.ndarray 
        The wav values that should be sent to the audio queue.
    rate : int
        The sample rate (in hz) of the output wav. ex 16000, 22000, 24050, 48000
    text : str
        The text that was converted to audio.
    is_normalize : bool
        When true the output will be normalized to be between [-1.0, 1.0]

    Returns:
    --------
    redis_response : int
        The index in the queue of the newly inserted item. 
        If value is not a positive integer then audio was not added to the queue. 
    """
    logger.info(f"TTS.push_to_queue()")
    logger.debug(f"TTS.push_to_queue({wav = }, {rate = }, {text = }, {is_normalize})")
    # If did not receive a valid list
    if (not isinstance(wav, list)) and (not isinstance(wav, np.ndarray)):
        logger.error("TTS.push_to_queue(): Received an empty wav")

    if (isinstance(wav, list)): 
        wav = np.array(wav)

    logger.debug(f"TTS.push_to_queue(): {len(wav) = }")
    logger.debug(f"TTS.push_to_queue(): {np.min(wav) =  :.3f}")
    logger.debug(f"TTS.push_to_queue(): {np.max(wav) =  :.3f}")
    logger.debug(f"TTS.push_to_queue(): Sending chunk to queue")
    logger.debug(f"TTS.push_to_queue(): {np.mean(wav) =  :.2f}")
    logger.debug(f"TTS.push_to_queue(): {type(json_wav) = }")
    #logger.debug(f"TTS.push_to_queue(): {json_wav[0:100] = }")
    
    # If should normalize
    if is_normalize:
        logger.debug(f"TTS.push_to_queue(): Normalizing wav")
        wav = voice_box.normalize(wav)
        logger.debug(f"TTS.push_to_queue(): {np.min(wav) =  :.3f}")
        logger.debug(f"TTS.push_to_queue(): {np.max(wav) =  :.3f}")
        logger.debug(f"TTS.push_to_queue(): {np.mean(wav) =  :.2f}")
        #logger.debug(f"TTS.push_to_queue(): {json_wav[0:100] = }")
    # If received a numpy array
    if isinstance(json_wav, np.ndarray):
        # Convert numpy array to normal python list
        json_wav = wav.astype(float).tolist()
    
    # Prepare redis message
    json_wav = wav.astype(float).tolist() 
    #redis_msg = json.dumps({
    #    "data": json_wav, 
    #    "rate": rate, 
    #    "text": text, 
    #    "priority": priority, 
    #    "request_time": (str(time.time()) if (type(request_time) is not str) else request_time),
    #    "response_time": str(time.time()),
    #    "message_group_id": None,
    #    "is_last_message": True
    #})
    # Push output to audio redis queue
    #redis_response = redis_conn.lpush("audio", redis_msg)

    #return redis_response

    return 1


def get_tts_with_retry(
        text:str,
        should_retry:bool=True,
        speed:float=1.01,
        voice_clone:str="major",
        is_log_retries:bool=True
):
    #logging.debug(f"main.tts({chunk = })")
    # Get wav from model
    wav, rate, wavs = voice_box.read_text(
                                text,
                                speed=speed
                            )
    
    # Check read speed
    read_speed, read_length = voice_box.get_read_speed(text, wav)
    logger.debug(f"TTS.process_text(): init {read_speed =  :.2f}")

    # If read speed is outside valid range
    if should_retry and (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
        logger.warning(f"TTS.process_text(): Bad read speed detected : {read_speed = :.2f}")

        
        # Save initial and subsequent reads along with their read speed
        all_reads = [
            {
                "read_speed" : read_speed,
                "wav" : wav
            }
        ]
        # Retry generating the output with different inputs
        # TODO: Change params on each try to give a better shot of producing valid output
        retry_attempt = 1
        retry_attempts = 3
        while (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
            logger.warning(f"TTS.process_text(): Retrying generation {retry_attempt}/{retry_attempts}")
                
            # Save first bad output
            session = SessionClass(expire_on_commit=True)
            session.add(Retry(
                voice_name=voice_clone,
                file_list=voice_box.speaker_wav,
                text=text,
                bad_output_wav=wav.tolist()
            ))
            session.commit()
            session.close()

            # Select a random set of samples from the fallback group
            if voice_clone in voice_catalogue:
                voice_box.speaker_wav = list(np.random.choice(voice_catalogue[voice_clone].fallback_file_list, 3, replace=False))
            else:
                voice_box.speaker_wav = list(np.random.choice(voice_clone, 1, replace=False))

            logger.warning(f"TTS.process_text(): Retying with : {voice_box.speaker_wav =}")
            #logger.warning(f"TTS.process_text(): {type(voice_box.speaker_wav) =}")
            #logger.warning(f"TTS.process_text(): {type(voice_box.speaker_wav[0]) =}")

            # Get audio output from TTS
            wav, rate, wavs = voice_box.read_text(
                                text,
                                speed=speed
                            )
            # Get read speed
            read_speed, read_length = voice_box.get_read_speed(text = text, wav = wav)
            # Save read
            all_reads.append(
                {
                    "read_speed" : read_speed,
                    "wav" : wav
                }
            )

            logger.debug(f"TTS.process_text(): {read_speed =  :.2f}")

            retry_attempt += 1
            if retry_attempt > retry_attempts: 
                break

        # Find the best read speed in each generated sample
        logger.debug(f"TTS.process_text(): Selecting best wav out of {len(all_reads)}")
        best_read_speed = -1
        best_read_index = -1
        for read_index in range(len(all_reads)):
            read_speed = all_reads[read_index]["read_speed"]
            if read_speed > best_read_speed and read_speed < read_speed_upper_threshold:
                best_read_speed = read_speed
                best_read_index = read_index
        # If found a wav that met criteria
        if best_read_index != -1:
            logger.debug(f"TTS.process_text(): Best wav on {read_index = } with {best_read_speed =}")
            wav = all_reads[best_read_index]["wav"]
        else:
            logger.debug(f"TTS.process_text(): None of the reads met criteria")

        # If still has a bad read speed
        if read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold:
            logger.error(f"TTS.process_text(): Could not generate valid audio for {text = }")

            #continue
    

    return wav, 24050


def process_text(
        text:str, 
        is_push_to_redis:bool=False,
        push_chunks:bool=True, 
        return_full:bool=False,
        speed:float=None,
        priority:str=None, 
        request_time:str=None,
        should_retry:bool=True,
        voice_clone:str="major"
    ):
    """
    Given a block of text, run the text through the tts model and send the resulting wav to the audio redis queue.

    Parameters:
    -----------
    test : str
        The input string that should be converted to spoken audio.
    push_chunks : bool
        A flag that specifies if the text should be broken into chunks or processed as one large text.
        With chunking each section generated will be sent to the redis queue while the next chunk is 
        processing.
    return_full : bool
        When true will return the full wav output to the client that made the request.
    priority : str
        The priority that the message is assigned in the initial api call. 
        Passed to the ui to control which audio clips get played at each moment. 
    request_time : str
        A timestamp of when the initial request that triggered the response to be generated was received.
        i.e. when the user speaks a command

    Returns:
    --------
    wav : list[float]
        A list of float values representing the generated wav form. The wav is single channel (mono), 32-bit float,
        24050hz sample rate. 
    """
    logger.info(f"TTS.process_text()")
    logger.debug(f"TTS.process_text({text = }, {push_chunks = }, {return_full = }, {speed = }, {voice_clone = })")

    # If given an empty string
    if not text or text.strip() == "":
        # Return nothing
        return []


    # Filter text
    for item in replace_list:
        text = text.replace(item[0], item[1])
    for item in remove_list:
        text = text.replace(item,"")
        
    # If given voice is in catalogue
    if voice_clone in voice_catalogue:
        logger.debug(f"TTS.process_text(): Using voice from catalogue {voice_clone} : {voice_catalogue[voice_clone].file_list}")
        # Set speaker wavs
        voice_box.speaker_wav = voice_catalogue[voice_clone].file_list

    # If given voice is a filename or list of filenames
    elif ".wav" in voice_clone:
        logger.debug(f"TTS.process_text(): Using custom voice file list {voice_clone}")
        voice_box.speaker_wav = voice_clone

    # Unknowen voice value
    else:
        logger.error(f"TTS.process_text(): Could not recognize {voice_clone = } defaulting to major")
        # Default to major voice
        voice_box.speaker_wav = voice_catalogue["major"].file_list
        voice_clone = "major"
    
    # If given voice is in catalogue
    if voice_clone in voice_catalogue:
        logger.debug(f"Reader.process_text(): Using voice from catalogue {voice_clone} : {voice_catalogue[voice_clone].file_list}")
        # Set speaker wavs
        voice_box.speaker_wav = voice_catalogue[voice_clone].file_list

    # If given voice is a filename or list of filenames
    elif ".wav" in voice_clone:
        logger.debug(f"Reader.process_text(): Using custom voice file list {voice_clone}")
        voice_box.speaker_wav = voice_clone

    # Unknowen voice value
    else:
        logger.error(f"Reader.process_text(): Could not recognize {voice_clone = } defaulting to major")
        # Default to major voice
        voice_box.speaker_wav = voice_catalogue["major"].file_list
        voice_clone = "major"
    

    logger.warning(f"Reader.process_text(): {voice_box.speaker_wav = }")
    #logger.warning(f"Reader.process_text(): {type(voice_box.speaker_wav) = }")
    #logger.warning(f"Reader.process_text(): {type(voice_box.speaker_wav[0]) = }")

    # If text contains silence tokens or is long text
    if "." in text or len(text) > 250:
        #logger.debug(f"TTS.main.process_text(): Interpret pauses")
        # Split on '.'
        #texts = text.split(".")

        # Split the texts into individual sentences
        texts = sent_tokenize(text)

        # Recombine short sentences
        combined_sentences = []
        current_entry = texts[0]
        for i in range(1, len(texts)):
            current_length = len(current_entry.split())
            next_length = len(texts[i].split())
            
            if current_length + next_length <= 10:  # Adjust the threshold as needed
                current_entry += " " + texts[i]
            else:
                combined_sentences.append(current_entry.strip())
                current_entry = texts[i]

        # Add the last entry if there is any
        if current_entry:
            combined_sentences.append(current_entry.strip())
        texts = combined_sentences

        #logger.debug(f"TTS.main.process_text(): {texts = }")
        
        # Save an id to identify the group of message sent to the ui
        group_id = uuid.uuid4()

        # Save the full wav of each chunk
        full_wav = []
        # For each chunk of text separated by silence
        for chunk_index in range(len(texts)):
            logger.debug ("-"*100)
            # Extract the chunk
            chunk = texts[chunk_index].strip()
            logger.debug(f"TTS.main.process_text(): Processing {chunk = }")
            # If chunk is empty
            if not chunk or chunk.strip() == "":
                # Skip
                logger.debug(f"TTS.main.process_text(Skip blank chunk)")
                continue

            # Determine if there should be a pause in the output
            pause_length = 0.0
            # If not the last chunk to speak 
            if chunk_index < len(texts) - 1:
                # If next chunk is empty (the result of splitting a ..)
                if texts[chunk_index + 1] == "":
                    # TODO: Move to config
                    pause_length = 0.15
                else:
                    pause_length = 0.05
            
            wav, rate = get_tts_with_retry(text=chunk,should_retry=should_retry, speed=speed, voice_clone=voice_clone)
            
            # If should add pause
            if pause_length > 0.0:
                # Append a pause to the wav
                wav = np.hstack((wav, np.zeros(int(24050 * pause_length), dtype=float)))
            
            # If should push individual chunks to redis
            if push_chunks:
                # Normalize individual wav
                wav = voice_box.normalize(wav)
                #wav_T = wav.astype(float).tolist()
                json_wav = wav.astype(float).tolist()
                #logging.debug(f"main.tts(): Received output from model: {type(wav) = }")
                # audio_queue.put(json.dumps({"data":json_wav, "rate": rate}))
                #logger.debug(f"TTS.main.process_text(): Sending chunk to queue")
                #logger.debug(f"TTS.main.process_text(): {len(wav) = }")
                #logger.debug(f"TTS.main.process_text(): {np.min(wav) = }")
                #logger.debug(f"TTS.main.process_text(): {np.max(wav) = }")
                #logger.debug(f"TTS.main.process_text(): {np.mean(wav) = }")
                #logger.debug(f"TTS.main.process_text(): {type(json_wav) = }")
                #logger.debug(f"TTS.main.process_text(): {json_wav[0:100] = }")
                
                # Check if is the last chunk being sent from the original text
                is_last_chunk = False
                # If is the last item in the list
                if (chunk_index == len(texts) - 1):
                    is_last_chunk = True
                # If is the second to last item in the list and the last item is an empty string
                if (chunk_index == len(texts) - 2) and (not texts[chunk_index + 1].strip()):
                    is_last_chunk = True

                response_time = time.time()
                #logger.debug(f"TTS.main.process_text(): {response_time =}")
                if is_push_to_redis:
                    redis_msg = json.dumps({
                                            "data": json_wav, 
                                            "rate": rate, 
                                            "text": chunk, 
                                            "priority": priority, 
                                            "request_time": request_time,
                                            "response_time": str(response_time),
                                            "message_group_id": str(group_id),
                                            "is_last_message": is_last_chunk
                                            })
                    #logger.debug(f"TTS.main.process_text(): {redis_msg = }")
                    try:
                        redis_response = redis_conn.lpush("audio", redis_msg)
                        logger.debug(f"TTS.main.process_text(): {redis_response = }")

                        # If did not get an appropriate 
                        if not redis_response or redis_response < 0:
                            logger.error("TTS.main.process_text(): Error sending audio to redis queue") 

                    except redis.exceptions.ConnectionError:
                        logger.error(f"TTS.main.process_text(): Error sending response to redis")

                gen_audio = GeneratedAudio(
                    model_name=voice_box.config["model"]["name"],
                    voice_name=voice_clone,
                    text=chunk,
                    wav=json_wav
                )
                # Push new voice to db
                session = SessionClass(expire_on_commit=True)
                session.add(gen_audio)
                session.commit()
                session.close()

            # Save chunk wav
            full_wav.append(wav)

        # Recombine wavs chunks
        wav = np.hstack(full_wav)
        # Normalize
        wav = voice_box.normalize(wav)

    # Else just run the full text
    else:
        logger.debug(f"TTS.main.process_text(): Running full message")
        wav, rate = get_tts_with_retry(text=text,should_retry=should_retry, speed=speed, voice_clone=voice_clone)
        
        logger.debug(f"TTS.main.process_text(): Normalizing")
        wav = voice_box.normalize(wav)
        json_wav = wav.astype(float).tolist()

        #logger.debug(f"TTS.main.process_text(): Sending chunk to queue")
        #logger.debug(f"TTS.main.process_text(): {len(wav) = }")
        #logger.debug(f"TTS.main.process_text(): {np.min(wav) = }")
        #logger.debug(f"TTS.main.process_text(): {np.max(wav) = }")
        #logger.debug(f"TTS.main.process_text(): {np.mean(wav) = }")
        #logger.debug(f"TTS.main.process_text(): {type(json_wav) = }")
        #logger.debug(f"TTS.main.process_text(): {json_wav[0:100] = }")


        # TODO: rename time to audio_queue time, add init_time
        response_time = time.time()
        #logger.debug(f"TTS.main.process_text(): {response_time = :.2f}")
        if is_push_to_redis:
            redis_msg = json.dumps({
                                    "data": json_wav, 
                                    "rate": rate, 
                                    "text": text, 
                                    "priority": priority, 
                                    "request_time": request_time,
                                    "response_time": str(response_time),
                                    "message_group_id": None,
                                    "is_last_message": True
                                    })
            #logger.debug(f"TTS.main.process_text(): {redis_msg = }")
            try:
                redis_response = redis_conn.lpush("audio", redis_msg)
                logger.debug(f"TTS.main.process_text(): {redis_response = }")
                # If did not get an appropriate 
                if not redis_response or redis_response < 0:
                    logger.error("TTS.main.process_text(): Error sending audio to redis queue") 
            except redis.exceptions.ConnectionError:
                logger.error(f"TTS.main.process_text(): Error sending response to redis")
        
        gen_audio = GeneratedAudio(
            model_name=voice_box.config["model"]["name"],
            voice_name=voice_clone,
            text=text,
            wav=json_wav
        )
        # Push new voice to db
        session = SessionClass(expire_on_commit=True)
        session.add(gen_audio)
        session.commit()
        session.close()

    return wav


def run_background_task(text: str):
    logger.debug("run_background_task(0)")
    thread = threading.Thread(target=process_text, args=(text))
    thread.start()


@app.post("/tts-stream/")
async def tts(message: Message):
    logger.debug(f"main.tts-stream({message = })")

    text = message.text
    run_background_task(text)
    
    return {"message": "Text transcribing"}


@app.post("/tts/")
async def tts(message: Message):
    logger.info(f"TTS.tts({message = })")
    # Extract text from message
    text = message.text
    speed = message.speed
    voice_clone = message.voice_clone

    # If given an empty string
    if not text or text.strip() == "":
        return {"message": "Error: Received an empty input string ", "wav": [], "rate" : str(24000)}

    else: 
        wav = process_text(text, priority=message.priority, request_time=message.time, speed=speed, voice_clone=voice_clone)

    logger.info(f"TTS.tts(): {type(wav)}")

    if isinstance(wav, NoneType) or (isinstance(wav, list) and len(wav) == 0):
        return {"message": "Error transcribing text", "wav": None, "rate" : str(24000)}

    # Return wav
    return {"message": "Text transcribed", "wav": wav.astype(float).tolist(), "rate" : str(24000)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
