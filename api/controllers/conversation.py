import os
import json
import time
import pickle
#import IPython
import logging
import requests
import numpy as np
from copy import copy
from datetime import datetime
from numpy.random import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import sys
sys.path.append("..") 
sys.path.append("../shared/") 
sys.path.append("../models/") 

from info import Info
from color import Color
from human import Human, Memory
from sentiment import Sentiment, SentimentScore
from models.comment import Comment
#from controllers.database_controller import DataBaseController

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def find_proper_nouns(sent):
    proper_nouns = []
    ner_tokens = preprocess(sent)
    for token in ner_tokens:
        if token[1] == "NNP":
            proper_nouns.append(token[0])
    # Convert list to string
    proper_nouns = " ".join(proper_nouns)
    return proper_nouns

class Conversation():
    def __init__(self, robot, is_debug=True):
        logging.info(f"{__class__.__name__}.__init__()")
        self.robot = robot        
        self.humans = []
        self.info = Info()
        self.chat_histories = {}
        self.chat_buffer_size = 4
        self.max_memory_insert_size = 3
        self.sentiment = Sentiment()
        self.is_debug = is_debug
        
        self.voice_ip = "192.168.1.120"
        self.voice_port = "8100"

        if "pygmalion" in robot.model_name.lower():
            self.model_type = "pygmalion"
        elif "mytho" in robot.model_name.lower():
            self.model_type = "mytho"
        elif "mistral" in robot.model_name.lower():
            self.model_type = "mistral"

        #self.dbc = DataBaseController()
        
    def __repr__(self):
        out_str =  f"Converstaion():\n"
        out_str +=  f"Robot: {self.robot.name}"
        out_str +=  f"chat_buffer_size: {self.chat_buffer_size}"
        out_str +=  f"Humans: {len(self.humans)}"
        for index in range(len(self.humans)):
            out_str +=  f"\t {index} : {self.humans[index]}"
        return out_str
    
    def save(self, filename="Conversation.p"):
        logging.info("{__class__.__name__}.save(): Save")

    def set_robot(self, robot):
        self.robot = robot
    
    def human_exists(self, name:str):
        for existing_human in self.humans:
            if existing_human.name == name:
                return True
        
    
    def get_human(self, name:str):
        for human in self.humans:
            if human.name == name:
                return human
        
    def add_human(self, human):
        if self.human_exists(human.name):
            logging.error(f"{Color.F_Red}Error: Human {human} already exists {Color.F_White}")
            return False
        self.humans.append(human)
        self.chat_histories[human.name] = ChatHistory(self.robot.name, human.name)
        
        return True
    
    def build_prompt(self,
                     commentor:str,
                     history_size:int=4
                    ):
        """
        Built an input prompt for the model given then 
        commentors history and the count of messages to include

        Parameters
        ----------
        commentor: str - The name of a user interacting with the system
        history_size: int - The number of recent messages to include in the prompt

        Returns
        -------
        prompt:str - A string containing the full prompt to send to the model encoder
        """
        logging.info(f"{__class__.__name__}.{__name__}(commentor={commentor}, history_size={history_size})")
        chat_history = copy(self.chat_histories[commentor])
        # If history too long
        if len(chat_history.dialogue) > history_size:
            # Select tail
            chat_history.dialogue = chat_history.dialogue[-history_size:]
        # Mix in human memories
        human = self.get_human(commentor)
        memory_insert_count = 0
        for index in range(len(human.positive_memories)):
            # Randomly insert memories
            if random() > 0.5:
                continue
            memory = human.positive_memories[index]
            # Check if memory is in chat history
            is_memory_included = False
            for comment in chat_history.dialogue:
                if comment.comment == memory.response.comment:
                    is_memory_included = True
            if not is_memory_included:
                #print(f"Inserting memory to prompt: memory = {memory}")
                chat_history.dialogue = [memory.prompt, memory.comment, memory.response] + chat_history.dialogue
            # Iterate memory insert count
            memory_insert_count += 1
            if memory_insert_count > self.max_memory_insert_size:
                break

        chat_history_prompt = chat_history.prompt()

        if len(chat_history_prompt.split(" ")) > 2048:
            logging.info(f"{__class__.__name__}.{__name__}(): Reducing size of chat history")
            chat_history_prompt = chat_history_prompt.split(" ")[0:2048]
            chat_history_prompt = " ".join(chat_history_prompt)

        if self.model_type == "pygmalion":
            prompt = f"{self.robot.name}'s Persona: {self.robot.persona}\n"
            prompt += "<START>\n"
            #prompt += "[DIALOGUE HISTORY]"       
            # Dialogue history
            prompt += chat_history_prompt
        
            # Randomly add things to prompt to steer conversation
            #if random() > 0.75:
            #    prompt += np.random.choice(self.robot.prompt_spices) + "\n"        
            #if random() > 0.5:
            #    prompt += f"Be {np.random.choice(self.robot.prompt_emotions)}.\n"
            
            # Robot name prompt
            prompt += f"{self.robot.name}:"

        elif self.model_type == "mytho":
            prompt = f"<System prompt/Character Card>\n"
            prompt += "### Instruction:\n"
            prompt += f"Write {self.robot.name}'s next reply in a chat between {commentor} and {self.robot.name}. Write a single reply only.\n"
            prompt += chat_history_prompt
            prompt += "### Response:"

        elif self.model_type == "mistral":
            prompt = f"<|im_start|>system\n"
            prompt += f"""You are {self.robot.name}, {self.robot.persona}\n"""
            prompt += chat_history_prompt
            prompt += f"{self.robot.name}:"

        
        return prompt
    
    
    def process_comment(self, 
                        commentor:str, 
                        comment:str,
                        response_length_modifier:int=0,
                        min_allowed_respose_len:int=2,
                        response_count:int=3,
                        is_speak_response:bool=False
                       ):
        """
        Given a comment from a user, add the comment to the log and 
        get a response from the robot.
        
        Parameters
        ----------
        commentor : string - The name of the user making the comment
        comment : string - The comment from the 
        """
        func_name = "process_comment"
        logging.info(f"{__class__.__name__}.{func_name}({commentor = }, {comment = }, {is_speak_response = })")
        start_time = time.time()
        if not self.human_exists(commentor):
            self.add_human(Human(commentor))        
        # Convenience variables
        human = self.get_human(commentor)
        chat_history = self.chat_histories[commentor]
        
        # Preprocess text
        comment = comment.replace("...", ".. . ")

        # Save comment
        user_comment = Comment(commentor, comment, self.sentiment.get_sentiment(comment))

        #self.dbc.save_comment(user_comment)
        #user_comment.save()
        
        # If there are any proper nouns in the text
        # search for a corresponding wiki page
        # add the summary to the context
        prompt_info = None
        proper_nouns = find_proper_nouns(comment)
        if proper_nouns:
            logging.info(f"{__class__.__name__}.{func_name}: Found proper nouns = {proper_nouns}")
            prompt_info = self.info.find_wiki_page(proper_nouns)
        
        # Get sentiment for the comment
        sentiment_dict = self.sentiment.get_sentiment(comment)
        sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
        # Create comment object
        # Save comment to chat history
        chat_history.add_comment(Comment(commentor, comment, sentiment))
        # Generate robot response
        prompt = self.build_prompt(commentor, self.chat_buffer_size)
        if prompt_info:
            prompt = prompt_info + "\n" + prompt
            logging.info(f"{__class__.__name__}.{func_name}(): Updating prompt with wiki data")

        logging.info(f"{__class__.__name__}().{func_name}: Sending prompt to robot:")
        logging.info("-"*100)
        logging.info(f"\n{Color.F_Cyan}{prompt}{Color.F_White}")
        logging.info("-"*100)

        # Randomize length of response
        min_len = 16 + int(16 * random()) + response_length_modifier
        max_len = 64 + int(64 * random()) + response_length_modifier

        # Generate output from the robot given the prompt
        outputs = self.robot.get_robot_response(commentor, prompt, min_len=min_len, max_len=max_len, response_count=response_count)

        if len(outputs) == 0:
            output = "Huh I don't know what to say"
        elif len(outputs) == 1:
            output = outputs[0]
        elif len(outputs) > 1:
            # Score each output
            output_scores = np.zeros(len(outputs))
            longest_output_count = 0
            longest_output_index = 0
            for index in range(len(outputs)):
                output = outputs[index]
                if len(output) > longest_output_count:
                    longest_output_count = len(output)
                    longest_output_index = index
                sentiment_dict = self.sentiment.get_sentiment(output)
                sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
                comment = Comment(self.robot.name, output, sentiment)
                score = sentiment_dict["positive_score"]
                output_scores[index] = score
                logging.info(f"{__class__.__name__}.{func_name}(): output[{index}] ")
                logging.info(f"{__class__.__name__}.{func_name}(): \t {len(output) = }")
                logging.info(f"{__class__.__name__}.{func_name}(): \t sentiment = {Color.F_Green}{int(100*round(sentiment_dict['positive_score'],2))} {Color.F_Red}{int(100*round(sentiment_dict['negative_score'],2))} {Color.F_White}")
                logging.info(f"{__class__.__name__}.{func_name}(): \t {comment.printf()}")
            # Give the longest response a boost in score
            output_scores[longest_output_index] += 0.25
            # Get the top scoring index
            top_index = np.argmax(output_scores)
            #logging.info(f"{__class__.__name__}.{func_name}(): Top comment [{top_index}]")
            # Pick the response to use
            output = outputs[top_index]

        # Text to speech output 
        wav, rate = None, None
        
        # If should speak response
        if is_speak_response:
            logging.info(f"{__class__.__name__}.{func_name}(): Reading response")
            wav = self.tts(output)
            #IPython.display.display(IPython.display.Audio(wav, rate=rate, autoplay=True))
        
        # Get sentiment for the comment
        sentiment_dict = self.sentiment.get_sentiment(output)
        sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
        # Create comment object
        self.chat_histories[commentor].add_comment(Comment(self.robot.name, output, sentiment))
        
        if len(chat_history.dialogue) > 2:
            logging.info(f"{__class__.__name__}.{func_name}(): last comment = {self.chat_histories[commentor].dialogue[-2].comment}")
            logging.info(f"{__class__.__name__}.{func_name}(): {self.chat_histories[commentor].dialogue[-2].sentiment}")
        # If has a complete set of prompt, generated response and user response
        if len(chat_history.dialogue) > 3: 
            # If the sentiment of the user response was positive
            if self.chat_histories[commentor].dialogue[-2].sentiment.sentiment == "positive":
                # Save positive interaction
                logging.info(f"{__class__.__name__}.{func_name}():  Got a positive response, saving memory")
                prompt = self.chat_histories[commentor].dialogue[-4]
                comment = self.chat_histories[commentor].dialogue[-3]
                response = self.chat_histories[commentor].dialogue[-2]
                human.add_positive_memory(Memory(prompt, comment, response))
            # If the sentiment of the user response was positive
            elif self.chat_histories[commentor].dialogue[-2].sentiment.sentiment == "negative":
                # Save negative interaction
                logging.info("f{__class__.__name__}.{func_name}():  Got a negative response, saving negative memory")
                prompt = self.chat_histories[commentor].dialogue[-4]
                comment = self.chat_histories[commentor].dialogue[-3]
                response = self.chat_histories[commentor].dialogue[-2]
                human.add_negative_memory(Memory(prompt, comment, response))

        runtime = time.time() - start_time
        tokens_per_sec = len(output.split(" ")) / runtime
        logging.info(f"{__class__.__name__}.{func_name}(): runtime = {runtime}")
        logging.info(f"{__class__.__name__}.{func_name}(): tokens_per_sec = {tokens_per_sec}")

        return output, wav


    def tts(self, text):
        
        payload = {'text': text, 'time': 'time', 'priority' : "100.0", "speed" : 1.27, "voice_clone" : "major"}
        
        start_time = time.time()
        r = requests.post(f"http://{self.voice_ip}:{self.voice_port}/tts", data=json.dumps(payload))

        if not r.status_code == 200:
            logging.error(f"{__class__.__name__}.tts(): Error getting tts response {r.status_code = }")
            return None
        if not "wav" in r.json():
            logging.error(f"{__class__.__name__}.tts(): Error missing wav in response {r.json() = }")
            return None
        
        # Extract wav
        wav = r.json()["wav"]

        # Profile
        end_time = time.time()
        run_time = end_time - start_time
        words_per_sec = len(text.split(" ")) / run_time
        logging.debug(f"{run_time = :.2f}s")
        logging.debug(f"{words_per_sec = :.2f}")
        logging.debug(f"Run_time ratio = {(len(wav) / 24000) / run_time :.2f}")

        return wav


class ChatHistory():
    def __init__(self, 
                 personA:str, 
                 personB:str,
                 use_cache:bool=True
                ):
        self.cache_filename = f"{personA}-{personB}_chat_history.p"

        if use_cache and os.path.isfile(self.cache_filename):
            logging.error(f"{__class__.__name__}.__init__(): Loading chat history from {self.cache_filename}")
            loaded = self.load()
            if not loaded:
                logging.error(f"{Color.F_Red}{__class__.__name__}.__init__(): Failed to load chat history {self.cache_filename}{Color.F_White}")
            else:
                logging.error(f"{__class__.__name__}.__init__(): Loaded chat history for {self.personA} -> {self.personB}")
        else:
            # Init a fresh chat
            self.personA = personA
            self.personB = personB
            self.dialogue = [] # list of Comment objects
            self.save()


    def save(self):
        pickle.dump({"personA":self.personA, "personB":self.personB, "dialogue":self.dialogue}, open(self.cache_filename, "wb"))

        return True

    def load(self):
        if not os.path.isfile(self.cache_filename):
            logging.warning(f"{__class__.__name__}.load(): File does not exist")
            return None
        
        saved = pickle.load(open(self.cache_filename, "rb"))

        self.personA = saved["personA"]
        self.personB = saved["personB"]
        self.dialogue = saved["dialogue"] # list of Comment objects

        return True
      
    def add_comment(self, comment):
        self.dialogue.append(comment)
        #self.save()
    
    def reset(self):
        self.dialogue = []
    
    def printf(self, count=None):
        if count:
            out_str = ""
            for comment in self.dialogue[-count:]:
                out_str += str(comment.printf())
                out_str += " \n"
        else:
            out_str = ""
            for comment in self.dialogue:
                out_str += str(comment.printf())
                out_str += " \n"
        return out_str
    
    def prompt(self):
        out_str = ""
        for comment in self.dialogue:
            #if len(comment.prompt()) > 512:
            #    out_str += comment.prompt()[0:512]
            #else:
            out_str += comment.prompt()
            out_str += " \n"
        return out_str
    