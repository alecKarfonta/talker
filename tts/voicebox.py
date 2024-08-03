import os
import json
import logging
from copy import copy
from time import time
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#from TTS.tts.models.xtts import Xtts
#from TTS.utils.manage import ModelManager
#from TTS.tts.configs.xtts_config import XttsConfig
#from TTS.api import TTS
from datasets import load_dataset
from pydub import AudioSegment
import numpy as np   
import soundfile as sf
from scipy.io.wavfile import read


from melo.api import TTS

from openvoice import se_extractor
from openvoice.api import ToneColorConverter



class VoiceBox():
    """
    Class used to allow easy access to various text to speech models. Can initialize models from several 
    different frameworks and allow a common interface for running inference on the model.
    """
    # TODO: Build a wrapper class around each model to avoid the branching logic
    def __init__(self, 
                logger:logging.Logger=None, 
                config:dict=None,
                config_filename:str="voicebox_config.json", 
                model_name:str=None,
                speaker_wav:str=None,
                synth_params:dict=None,
                silence_filter_params:dict=None,
                pre_init:bool=True,
                is_persist:bool=True
                ):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("DEBUG")
        self.logger.debug(f"{__class__.__name__}.init()")
        # Show dep versions
        expose(prefix = f"{__class__.__name__}.init(): ")

        if config:
            self.config = config
        elif config_filename:
            self.config = self.load_config(config_filename)
        else:
            self.logger.error(f"{__class__.__name__}.init(): Must provide either a config dict or filename")
            return None
        
        self.logger.debug(f"{__class__.__name__}.init(): Config = {json.dumps(self.config, indent=3)}")

        # Load overrides
        # Model name
        if model_name:
            self.model_name = model_name
            self.config["model"]["name"] = synth_params
        else:
            self.model_name = self.config["model"]["name"]
        # Speaker definition
        if speaker_wav:
            self.speaker_wav = speaker_wav
            self.config["vocoder"]["speaker_wav"] = synth_params
        else:
            self.speaker_wav = self.config["vocoder"]["speaker_wav"]
        # Synthesizer params
        if synth_params:
            self.synth_params = synth_params
            self.config["synth_params"] = synth_params
        else:
            self.synth_params = self.config["synth_params"]
        # Silence filter params
        if silence_filter_params:
            self.silence_filter_params = silence_filter_params
            self.config["silence_filter_params"] = silence_filter_params
        else:
            self.silence_filter_params = self.config["silence_filter_params"]       

        # Save model variable
        #self.tacotron2 = None
        #self.hifi_gan = None
        self.model = None
        # If should pre-initialize the model
        if pre_init:
            self.init_model()

        self.start_time = time()        
        self.is_persist = is_persist

        self.logger.debug(f"{__class__.__name__}.init(): Complete at {self.start_time}")


    def load_config(self, config_filename:str="voicebox_config.json"):
        # Load config file
        #try:
        config = json.load(open(config_filename, "r"))
        #except:
        #    logging.error(f"{__class__.__name__}.init(): Could not read config file at {config_filename}")
        #    return None

        return config


    def save_config(self, config_filename:str="voicebox_config.json"):
        with open(config_filename, "w") as config_file:
            config_file.write(json.dumps(self.config))


    def init_model(self, model_name:str=None):
        """
        Init the listener model. 
        """

        model = None

        if not model_name:
            model_name = self.model_name
        else:
            self.model_name = model_name

        self.logger.debug(f"{__class__.__name__}.init_model(): {self.model_name = }")

        if self.model_name == "openvoice":
            ckpt_converter = 'checkpoints_v2/converter'
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            device = "cpu"
            output_dir = 'outputs_v2'
            speed = 1.0

            tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)


        # If using fine-tuned voice model
        elif self.model_name == "xtts_v1.1":
            from TTS.tts.models.xtts import Xtts
            from TTS.tts.configs.xtts_config import XttsConfig
            self.logger.debug(f"{__class__.__name__}.init_model(): Init xtts model")
            # Download model
            #ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v1.1")
            # Set path to model files
            # TODO: Move to config file
            model_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/model.pth"
            config_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/config.json"
            vocab_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/vocab.json" 
            # Init config 
            self.model_config = XttsConfig()
            self.model_config.load_json(config_path)
            # Init model
            self.model = Xtts.init_from_config(self.model_config)
            self.model.load_checkpoint(
                self.model_config,
                checkpoint_path=model_path,
                vocab_path=vocab_path,
                eval=True,
                use_deepspeed=False
            )
            # If is using the gpu
            if not self.config["model"]["device"] == "cpu":
                self.logger.debug(f"{__class__.__name__}.init_model(): Sending model to gpu")
                self.model.to(self.config["model"]["device"])

        elif self.model_name == "xtts_v2":
            from TTS.tts.models.xtts import Xtts
            #from TTS.utils.manage import ModelManager
            from TTS.tts.configs.xtts_config import XttsConfig
            #from TTS.api import TTS
            self.logger.debug(f"{__class__.__name__}.init_model(): Init xtts model")
            # Download model
            #ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
            # Set path to model files
            # TODO: Move to config file
            model_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2/model.pth"
            config_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2/config.json"
            vocab_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2/vocab.json" 
            
            # Init config 
            self.model_config = XttsConfig()
            self.model_config.load_json(config_path)
            # Init model
            self.model = Xtts.init_from_config(self.model_config)
            self.model.load_checkpoint(
                self.model_config,
                checkpoint_path=model_path,
                vocab_path=vocab_path,
                eval=True,
                use_deepspeed=self.config["model"]["is_deepspeed"]
            )
            # If is using the gpu
            if not self.config["model"]["device"] == "cpu":
                self.logger.debug(f"{__class__.__name__}.init_model(): Sending model to gpu")
                self.model.to(self.config["model"]["device"])

        elif self.model_name == "tortoise":
            from tortoise.api import TextToSpeech
            self.model = TextToSpeech(use_deepspeed=True, kv_cache=True, device=self.config["model"]["device"])

        # Else using Tacotron model
        elif self.model_name == "hf-tacotron2":
            from speechbrain.pretrained import Tacotron2
            from speechbrain.pretrained import HIFIGAN
            # Read config parameters
            model_config = self.config["models"][model_name]
            model_name = self.config["model"]["name"]
            vocalizer_name = self.config["vocoder"]["name"]
            device = model_config["model"]["device"]
            
            self.logger.debug(f"{__class__.__name__}.init_model(): Loading {model_name} model")

            # Init text to speach model
            self.tacotron2 = Tacotron2.from_hparams(source=model_name, savedir="tmpdir_tts", run_opts={"device":device})
            self.hifi_gan = HIFIGAN.from_hparams(source=vocalizer_name, savedir="tmpdir_vocoder", run_opts={"device":device})

        # Else using a t5 model
        elif self.model_name == "speecht5_tts":
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            speaker_id = 1300
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(self.embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

        #elif model_name.startswith("tts_models/"):
        #    self.tts_model = TTS(model_name) #.to(device)
        # Else unrecognized model
        else:
            self.logger.error(f"{__class__.__name__}.init_model(): Unknown model name {model_name}")
        self.logger.debug(f"{__class__.__name__}.init_model(): Loading model")
    

    def tts(self, text:str):
        """
        Given a piece of text use the appropriate model to convert the text to a spoken audio wav.

        Parameters:
        -----------
        text : str - An input string that should be converted to audio

        Returns:
        --------
        wav : np.ndarray[float] - An array of float values representing the audio signal
        samplerate : int - An integer giving the sample rate to use when playing the audio
        """
        #self.logger.debug(f"{__class__.__name__}.tts({text = })")
        #self.logger.debug(f"{__class__.__name__}.tts({self.model_name = })")

        # If given empty string
        if not text or text.strip() == "":
            return [], 24050

        if self.model_name == "xtts_v1.1" or self.model_name == "xtts_v2":
            
            #self.logger.debug(f"{__class__.__name__}.tts(): Using xtts model with")
            #self.logger.debug(f"{__class__.__name__}.tts(): params = {self.config['synth_params']}, {self.speaker_wav = }")
            outputs = self.model.synthesize(
                text,
                self.model_config,
                language="en",
                speaker_wav=self.speaker_wav,
                **self.synth_params
            )

            wav = outputs["wav"]
            
            #print(f"voicebox.tts(): {len(wav) = }")
            #print(f"voicebox.tts(): {type(wav) = }")
            #print(f"voicebox.tts(): {type(wav[0]) = }")
            
            return wav, 24050
        
        elif self.model_name == "tortoise":
            self.logger.debug(f"{__class__.__name__}.tts(): Running tortoise model")
            from tortoise.utils.audio import load_audio
            reference_clips = [load_audio(p, 22050) for p in self.speaker_wav]
            #self.logger.debug(f"{__class__.__name__}.tts(): Using reference clips: {self.speaker_wav}")

            pcm_audio = self.model.tts_with_preset(text, voice_samples=reference_clips, preset=self.config["model"]["preset"])

            audio_parts = []
            for candidate_idx, audio in enumerate([pcm_audio]):
                audio = audio.squeeze(0).cpu()
                if candidate_idx == 0:
                    audio_parts.append(audio) 
            audio = torch.cat(audio_parts, dim=-1)

            self.logger.debug(f"{__class__.__name__}.tts(): {type(audio) = }")
            self.logger.debug(f"{__class__.__name__}.tts(): {audio.shape = }")

            wav = audio.numpy()[0]

            return wav, 24050

        # IF using Tacotron model
        elif self.model_name == "hf-tacotron2":
            self.logger.debug(f"{__class__.__name__}.tts(): Running tacotron model {self.model_name = }")
            # Running the TTS
            mel_output, mel_length, alignment = self.tacotron2.encode_text(text)
            self.logger.debug(f"{__class__.__name__}.tts(): Got model output {mel_length =}")
            # Running Vocoder (spectrogram-to-waveform)
            wav = self.hifi_gan.decode_batch(mel_output)
            wav = wav.squeeze(1)
            #print (f"{type(wav) = }")
            return wav, 16000
        
        # If using T5 Model
        elif self.model_name == "speecht5_tts":
            self.logger.debug(f"{__class__.__name__}.tts(): Running t5 model {self.model_name = }")
            inputs = self.processor(text=text, return_tensors="pt")
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            return speech.numpy(), 16000

        # Disabled
        elif False and self.model_name.startswith("tts_models/"):
            self.logger.debug(f"{__class__.__name__}.tts(): Running tts model {self.model_name = }")
            tts_args = {}
            
            if self.tts_model.is_multi_lingual:
                tts_args["language"] = "en"
            
            if self.speaker_wav:
                tts_args["speaker_wav"] = self.speaker_wav
            else:
                #if self.tts_model.tts.is_multi_speakers and not self.speaker_wav:
                tts_args["speaker"] = self.tts_model.speakers[0]
            
            self.logger.debug(f"{__class__.__name__}.tts(): {tts_args =}")
            self.logger.debug(f"{__class__.__name__}.tts(): Calling model")

            wav = self.tts_model.tts(
                        text=text,
                        **tts_args
                )
            self.logger.debug(f"{__class__.__name__}.tts(): Got model results {len(wav) = }")
            return wav, 16000
        else:
            self.logger.debug(f"{__class__.__name__}.read_text(): Unknown model name {self.model_name}")
        
        return None, None


    @classmethod 
    def normalize(self, arr:np.ndarray) -> np.ndarray:
        # Min max normalization
        #min_val = np.min(arr)
        #max_val = np.max(arr)
        #print (f"normalizing between {arr[0:10] = } ")
        #print (f"normalizing between {min_val} - {max_val} ")
        #scaled_arr = (arr - min_val) / (max_val - min_val)
        #print (f"normalizing between {scaled_arr[0:10] = } ")
        if isinstance(arr, list) and len(arr) == 0:
            return arr

        if isinstance(arr, np.ndarray) and arr.shape[0] == 0:
            return arr
        # Scaled normalization
        #self.logger.debug(f"{__class__.__name__}.normalize(): normalizing with  {np.max(arr) = } ")
        scaled_arr = arr * (1.0 / np.max(arr))
        return scaled_arr


    def normalize_min_max(self, arr:np.ndarray):
        # Normalize the array to [0-1] using the min max values
        return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))
    


    def chunk_text(text:str, max_words:int=30):
        # If text is long        
        if word_tokenize(text) > max_words:

            # Try to split on sentences
            text_split = sent_tokenize(text)


    def read_text(
            self, 
            text:str,
            concat_delay:int=12500,
            speed:float=None
        ):
        """
        Given a block of text use the tts model to convert to audio. Chunk the text file if necessary.
        
        Parameters:
        -----------
            text : str - A potentially large block of input text that should be converted to audio
        Returns:
        --------
            wav : np.ndarray[float] - An array of float values representing the audio signal
            samplerate : int - An integer giving the sample rate to use when playing the audio
        """
        #self.logger.info(f"{__class__.__name__}.read_text()")
        #self.logger.debug(f"{__class__.__name__}.read_text({text = }")
        
        # If given empty string
        if not text or text.strip() == "":
            return [], 24050
        
        start_time = time()
        #logging.info(f"{__class__.__name__}.{__name__}(text={text}, rate_modifier={rate_modifier}, is_say={is_say})")
        wav = None
        wavs = None
        sample_rate = None
        # If text is long        
        if len(text) > 250:
            self.logger.debug(f"VoiceBox.read_text(): Splitting long text {text}")
            self.logger.debug(f"{__class__.__name__}.read_text(): Splitting text")
            #text_split = text.split(".") 
            # Split on sentences
            text_split = sent_tokenize(text)
            new_text_split =  []
            for sentence in text_split:
                # If the sentence is short enough
                if len(word_tokenize(sentence)) < 30:
                    # Add it to the list
                    new_text_split.append(sentence)
                    continue
                # Else need to break the sentence down in some way
                # If has a newline
                if "\\n" in sentence:
                    # Split on that
                    sentences = sentence.split("\\n")
                    new_text_split.extend(sentences)
                    continue
                
                
                #for new_sentence in sentences:
            text_split = new_text_split
            # Save each sentence wav
            wavs = []
            # For each sentence
            for sentence_index in range(len(text_split)):
                sentence = text_split[sentence_index]
                #self.logger.debug(f"{__class__.__name__}.read_text(): Reading {sentence =}")
                sentence = sentence.strip()
                #wav, rate = robot.read_response(sentence.strip())
                if len(sentence) < 2:
                    self.logger.debug(f"VoiceBox.read_text(): Skipping short text {len(sentence) = }")
                    #self.logger ("Skipping")
                    continue
                #try:
                    
                #self.logger.debug(f"{__class__.__name__}.read_text(): Calling self.tts")
                wav, sample_rate = self.tts(sentence)
                #self.logger.debug(f"{__class__.__name__}.read_text(): Got results from self.tts")

                # Save wav
                wavs.append(wav)

                if sentence_index < len(text_split) - 1:
                    wavs.append(np.zeros(concat_delay))
                    
                #except Exception as ex:
                #    logging.error(f"{__class__.__name__}.read_text(): Failed to read text = {sentence = } ")
                #    logging.error(f"{__class__.__name__}.read_text(): {ex = } ")

            # Stack the wavs into a single wav
            if (isinstance(wav, np.ndarray)):
                #self.logger.debug(f"{__class__.__name__}.read_text(): Output is a numpy array {wav.shape }")
                wav = np.hstack(wavs)
            elif (isinstance(wav, torch.Tensor)):
                wav = torch.hstack(wavs)


        else:
            #self.logger.debug(f"{__class__.__name__}.read_text(): Running full text {text}")
            #try:
            wav, sample_rate = self.tts(text)
            #self.logger.debug(f"{__class__.__name__}.read_text(): Got model response: {len(wav) = }")

        
        #wav = wav.squeeze(1)
        #except:
        #    logging.error(f"{Color.F_Red}{__class__.__name__}.read_text(): Failed to read text = {text}{Color.F_White}")

        # Normalize
        #wav = self.normalize(wav)

        # TODO: Check initial read speed and only ever apply one speed up
        init_read_speed, init_read_length = self.get_read_speed(text, wav)
        self.logger.debug(f"{__class__.__name__}.read_text(): {init_read_speed = :.2f}")

        #wav, read_speed = self.adjust_read_speed(wav, text, init_read_speed)

        # Determine playback speed to hit target read speed (words per second)
        volume = None
        # If not given a specific speed
        if not speed:
            # Use speed from config
            speed = self.config["vocoder"]["speed_up"]
        
        #if init_read_speed < upper
        
        # Set the speed adjustment based on initial read speed
        #if init_read_speed < 0.8:
        #    speed = speed + (speed * 0.15)
        #elif init_read_speed < 0.9:
        #    speed = speed + (speed * 0.1)
        #if init_read_speed < 1.0:
        #    speed = speed + (speed * 0.05)


        # TODO: Return the initial wav file before post processing to evaluate the effect on audio quality
        #orig_wav = copy(wav)
        # Apply post processing
        wav = self.apply_post_processing(wav, speed=speed, volume=volume)

        # If ran on gpu
        #if "cuda" in self.config["model"]["device"]:
        #    # Send array back to cpu before returning
        #    wav = wav.cpu()
        
        
        runtime = time() - start_time
        self.logger.debug(f"{__class__.__name__}.read_text(): runtime = {runtime : .2f}")
        return wav, sample_rate, wavs


    def get_read_speed(self, text, wav, sample_rate=24050):
        if len(wav) == 0:
            return -1, -1
        # Measure the spoken words per second in the audio
        words = nltk.tokenize.word_tokenize(text)
        word_count = 0
        for word in words:
            if word in [","]:
                continue
            word_count += 1
        audio_length = len(wav) / sample_rate
        words_per_second = word_count / audio_length
        return words_per_second, audio_length


    def adjust_read_speed(self, wav, text:str, read_speed:float):
        """
        Given a wav float array, try to normalize the read speed to a target value.
        Measure the words per second spoken in the wav. Then based on the value
        apply a sliding scale of adjustments to get the audio closer to a normalized 
        read speed.

        Parameters:
        -----------
        wav : list[float|np.ndarray]
            A float array representing an audio waveform.
        text : str
            The text that the TTS model used to generate the waveform.
        read_speed : float
            The initial measured read_speed.
        
        Returns:
        --------
        wav : list[float|np.ndarray]
            The waveform with the applied speed up, if any were triggered.
        read_speed : float
            The new read speed with the speed up.
        """
        #logger.info(f"Reader.adjust_read_speed()")
        #logger.debug(f"Reader.adjust_read_speed(wav, {text = }, {read_speed = })")
        self.logger.debug(f"Reader.adjust_read_speed(wav, text, {read_speed = })")
        # If read speed is close to target
        # Apply a sliding scale speed up
        initial_read_speed = read_speed
        # If read speed is close to target
        # Apply a sliding scale speed up
        if read_speed > 1.35 and read_speed < 1.4:
            self.logger.warning(f"Reader.adjust_read_speed(): Very Slow read speed detected")
            # Apply a another speed up
            wav = self.apply_speed_up(wav, 1.15)
            read_speed, read_length = self.get_read_speed(text, wav)
            self.logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

        if read_speed > 1.4 and read_speed < 1.5:
            self.logger.warning(f"Reader.adjust_read_speed(): Slow read speed detected")
            # Apply a another speed up
            wav = self.apply_speed_up(wav, 1.1)
            read_speed, read_length = self.get_read_speed(text, wav)
            self.logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

        if read_speed > 1.5 and read_speed < 1.55:
            self.logger.warning(f"Reader.adjust_read_speed(): Slightly slow read speed detected")
            # Apply a another speed up
            wav = self.apply_speed_up(wav, 1.05)
            read_speed, read_length = self.get_read_speed(text, wav)
            self.logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

        #if read_speed > read_speed_upper_threshold:
        #    logger.warning(f"Reader.adjust_read_speed(): Fast read speed detected")
        #    #wav = voice.apply_speed_up(wav, 1.32)
        #if read_speed != initial_read_speed:
        #    logger.debug(f"Reader.adjust_read_speed(): Adjusted read speed from {initial_read_speed :.2f} to {read_speed :.2f}")

        return wav, read_speed


    def apply_speed_up(self, wav, speed):
        """
        Given a wav file perform a speed up to increase the spoken words per second.        
        """
        self.logger.debug(f"{__class__.__name__}.apply_speed_up(wav, {speed = })")
        sf.write("cache/test.wav", wav, 24050)
        sound = AudioSegment.from_file("cache/test.wav", format = "wav")
        #self.logger.debug(f"{__class__.__name__}.apply_speed_up(): Applying speed up")
        sound = sound.speedup(speed, 150, 25)
        sound = sound.strip_silence(**self.silence_filter_params)
        wav = np.array(sound.get_array_of_samples())
        return wav


    def apply_post_processing(self, wav, speed:float=None, volume:float=None, sample_rate=24050):
        #self.logger.info(f"{__class__.__name__}.apply_post_processing()")
        #self.logger.debug(f"{__class__.__name__}.apply_post_processing(wav, {speed = }, {volume = })")

        # If has some kind of post processing to apply
        if (self.config["vocoder"]["speed_up"] != 1.0 or speed) or \
                self.config["vocoder"]["apply_dynamic_compression"] or \
                self.config["vocoder"]["apply_low_pass"] or \
                self.config["vocoder"]["apply_high_pass"]:
                #print ("Applying post processing")
                # Apply post processing
                try:
                    os.mkdir("cache")
                except FileExistsError:
                    pass
                #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Applying post processing")
                sf.write("cache/test.wav", wav, sample_rate)
                sound = AudioSegment.from_file("cache/test.wav", format = "wav")

                if volume:
                    #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Adjusting volume")
                    sound = sound + volume

                #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Striping silence")
                sound = sound.strip_silence(**self.silence_filter_params)

                # Post processing
                if (self.config["vocoder"]["speed_up"] != 1.0 or speed) and sound.duration_seconds > 0.5:
                    # Speed adjustment
                    #self.logger.debug(f"{__class__.__name__}.tts(): Applying speed up")
                    if speed:
                        #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Applying speed up: {speed}")
                        sound = sound.speedup(speed, 150, 25)
                    else:
                        self.logger.debug(f'{__class__.__name__}.apply_post_processing(): Applying speed up: {self.config["vocoder"]["speed_up"]}')
                        sound = sound.speedup(self.config["vocoder"]["speed_up"], 150, 25)
                if self.config["vocoder"]["apply_dynamic_compression"]:
                    # Dynamic range compression
                    #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Applying dynamic range compression")
                    sound = sound.compress_dynamic_range(
                                threshold = self.config["vocoder"]["threshold"], # Default -20
                                ratio = self.config["vocoder"]["ratio"], # Default 4.0
                                attack = self.config["vocoder"]["attack"], # Default 5.0
                                release = self.config["vocoder"]["release"], # Default 50.0
                    )
                if self.config["vocoder"]["apply_low_pass"]:
                    # Low pass filter
                    #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Applying low pass filter")
                    try:
                        sound = sound.low_pass_filter(
                                    cutoff = self.config["vocoder"]["low_pass_cutoff"],
                        )
                    except IndexError:
                        self.logger.error("low_pass_filter failed")
                
                if self.config["vocoder"]["apply_high_pass"]:
                    # High pass filter
                    #self.logger.debug(f"{__class__.__name__}.apply_post_processing(): Applying high pass filter")
                    try:
                        sound = sound.high_pass_filter(
                                    cutoff = self.config["vocoder"]["high_pass_cutoff"],
                        )
                    except IndexError:
                        self.logger.error("{__class__.__name__}.apply_post_processing(): high_pass_filter failed")

                #so = sound.speedup(self.config["vocoder"]["speed_up"], 1.5, 150)
                wav = np.array(sound.get_array_of_samples())
                #wav = sound.get_array_of_samples()
        else:
            pass
        
        return wav


def expose(prefix:str):
    import torch
    print (f"{prefix}{torch.__version__ = }")
    import transformers
    print (f"{prefix}{transformers.__version__ = }")
    import nltk
    print (f"{prefix}{nltk.__version__ = }")
    #import speechbrain
    #print (f"{prefix}{speechbrain.__version__ = }")
    import TTS
    print (f"{prefix}{TTS.__version__ = }")
    import datasets
    print (f"{prefix}{datasets.__version__ = }")