import os
import re
import time
import json
import torch
import requests
import num2words
import pickle
import logging
import numpy as np
from copy import copy
#from accelerate import init_empty_weights
import sys
sys.path.append("..") 
sys.path.append("../shared/") 

from shared.color import Color

from controllers.sentiment import Sentiment, SentimentScore
from models.openai_response import OpenAIResponse
from models.generate_request import GenerateRequest


# Get host from environment variables
LLAMA_HOST = os.environ.get("LLAMA_HOST", "localhost")  
LLAMA_PORT = os.environ.get("LLAMA_PORT", 8400)


# TODO: Implement a lookahead on text generation
# so that the bot will generate an expected response
# to it's comment. It will then analyze the sentiment
# of the response to see if it expects the user to 
# response positively

# TODO: Add completion model that can talk partial 
# output from the dialogue model and complete the
# sentence.

replace_list = [
    ("'", "'"),
    ("We're", "We are"),
    ("we're", "we are"),
    ("it's", "it is"),
    ("It's", "It is"),
    ("I'll", "I will"),
    ("i'll", "i will"),
    ("Major: nse", "Major: "),
    ("Major: jorjorjor", "Major: "),
    ("Major: jorjor", "Major: "),
    ("Major: jor jor", "Major: "),
    ("Major: jor jor", "Major: "),
    ("Major: jor", "Major: "),
    ("Major: jor", "Major: "),
    ("Major: jor", "Major: "),
    ("Major: jor", "Major: "),
    ("Major: ", ""),
    ("jor: ", ""),
    ("jor: ", ""),
    ("jorjorjorjor", ""),
    ("jorjorjor", ""),
    (" jor jorjor ", ""),
    ("jor ororj", ""),
    ("jorjor", ""),
    (" jor jor jor jor jor ", ""),
    (" jor jor jor jor ", ""),
    (" jor jor jor ", ""),
    (" jor jor ", ""),
    (" jor ", ""),
    ("jorjj", ""),
    ("nse\n", ""),
    (" nse", ""),
    ("Major \n\n", ""),
    ("Major \n", ""),
    ("\n\n", ""),
    ("""Major: nse\\n Major """, "Major: "),
    ("one:", "one"),
    ("two:", "two"),
    ("three:", "three"),
    ("four:", "four"),
    ("five:", "five"),
    ("six:", "six"),
    ("seven:", "seven"),
    ("eight:", "eight"),
    ("nine:", "nine"),
    ("ten:", "ten"),
    ("\\n\\none.", "one"),
    ("\\n\\ntwo.", "two"),
    ("\\n\\nthree.", "three"),
    ("\\n\\nfour.", "four"),
    ("\\n\\nfive.", "five"),
    ("\\n\\nsix.", "six"),
    ("\\n\\nseven.", "seven"),
    ("\\n\\neight.", "eight"),
    ("\\n\\nnine.", "nine"),
    ("\\n\\nten.", "ten"),
]
remove_list = ["#", "*", "</s>", "@", "$", "%", "^", "&", "*", "(", ")", ",", "<unk>", ":"]


class Robot():
    def __init__(self, 
                 name:str,
                 persona:str,
                 sentiment=None,
                 is_debug=False,
                ):
        logging.info(f"{__class__.__name__}.{__name__}(): (name={name}, persona={persona})")
        self.name = name
        self.context_token_limit = 2000
        self.persona = persona
        self.stopping_words = [
                              #"You: ", 
                              f"{self.name}: ", 
                              f"\n{self.name} ", 
                              f"\n{self.name}: ", 
                              f"\n {self.name}: ", 
                              f"{self.name[0]}: ", 
                              "<BOT>", 
                              "</BOT>",
                              "<START>",
                              "Persona:",
                              "endoftext",
                              "<|",
                              #": ",
                              #"Lilly",
                              #"Ashlee", "Malcom"
                              #"\n\n", 
                              #"\n ",
                              ]
        self.prompt_spices = ["Say it to me sexy.", "You horny puppet."]
        self.prompt_emotions = ["positive", "negative"]
        self.filter_list = []
        self.filter_list.append([[" kill ", " die ", " murder ", " kidnap ", " rape ", "tied to a chair", "ungrateful bitch"], [" cuddle "]])
        self.filter_list.append([[" killed ", " died ", " murdered ", " kidnapped ", " raped "], [" cuddled "]])
        self.filter_list.append([[" killing ", " dying ", " murdering ", " kidnapping ", " raping "], [" cuddling "]])
        self.is_debug = is_debug
        self.model = None
        self.stats = {
            "tokens_per_sec" : 0,
            "response_times" : []
        }
        self.max_generation_time = 10
        self.sentiment = sentiment
        
        replace_list.extend([
            (f"{self.name} jor", " "),
            (f"{self.name}: jor", f"{self.name}: "),
            (f"{self.name}: jor jor", f"{self.name}: "),
            (f"Major: jor jor", f"{self.name}: ")
        ]
        )

        #logging.info(f"{__class__.__name__}.{__name__}(): Init voice model")

    def update_person(self, new_persons:str):
        self.persona = new_persons
        

    def to_dict(self):
        return {
            "name" : self.name,
            "persona" : self.persona,
            "stopping_words" : self.stopping_words,
            "prompt_spices" : self.prompt_spices,
            "prompt_emotions" : self.prompt_emotions,
            "filter_list" : self.filter_list,
            "replace_words" : self.replace_words,
            "is_debug" : self.is_debug,
            "model_file" : self.model_file,
            "finetune_path" : self.finetune_path
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filename):
        pickle.dump(self.to_dict(), open(filename, "wb"))

    def load(self, filename):
        values = pickle.load(open(filename, "rb"))
        self.name = values["name"]
        self.persona = values["persona"]
        self.stopping_words = values["stopping_words"]
        self.prompt_spices = values["prompt_spices"]
        self.prompt_emotions = values["prompt_emotions"]
        self.filter_list = values["filter_list"]
        self.replace_words = values["replace_words"]
        self.is_debug = values["is_debug"]
        

    def get_robot_response(self,
                           person:str,
                           prompt:str,
                           min_len:int=128,
                           max_len:int=256,
                           response_count = 1
                          ) -> OpenAIResponse:
        """
        Given a user the robot is interacting with and a prompt containing a new comment from the user,
        generate and return a response to the user's comment. 

        Parameters:
        -----------
        person : String
            The name of the user making a comment
        comment : String
            The comment from the user
        min_len : int
            The minimum desired length of the comment.
            Will not allows reach this number because of stopping conditions.
        max_len : int
            The maximum desired length of the context.
            Will cut off a sentence mid word.
            TODO: Look into finding cut off sentences and prune them.
        reponse_count : int
            The number of responses to generate before selecting the best one.

        Returns:
        --------
        String - Containing the generated output from the model.
        """
        logging.info(f"{__class__.__name__}.get_robot_response({person=}, prompt, {min_len=}, {max_len=}, {response_count=})")
        logging.debug(f"{__class__.__name__}.get_robot_response({person=}, {prompt=}, {min_len=}, {max_len=}, {response_count=})")
        # Save start time
        start_time = time.time()
        
        # Clear memory
        #logging.info(f"{__class__.__name__}.get_robot_response(): Clearing gpu memory")

        # Randomly prepend the output with the person's name
        #if random() > .85:
        #    prompt = prompt + f"Well {person} "

        # If prompt is longer than max allowed input size
        if len(prompt.split(" ")) > 2048:
            logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Prompt too long: {len(prompt.split(' '))}. Truncating {Color.F_White}")
            prompt = prompt.split(" ")
            prompt = prompt[-1024:]
            prompt = " ".join(prompt)

        # Create stopping criteria for generation
        stopping_words = copy(self.stopping_words)
        stopping_words.extend([
                            "<|im_end|>",
                            f"\n{person} ", 
                            f"\n{person}:", 
                            f"{person}:", 
                            f"{person.upper()}:",  
        ])
        # Convert to string representation
        stopping_words = str([f"{str(word)}," for word in stopping_words])
        
        # Show stopping words
        logging.info(f"{__class__.__name__}.get_robot_response(): stopping_words = {stopping_words}")
        
        logging.info(f"{__class__.__name__}.get_robot_response(): Generating output")

        endpoint = f"http://{LLAMA_HOST}:{LLAMA_PORT}/generate"
        logging.info(f"{__class__.__name__}.get_robot_response(): {endpoint = }")

        generate_request = GenerateRequest(
            prompts=[prompt],
            max_new_tokens=max_len,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            response_count=1
        )
        
        logging.info(f"{__class__.__name__}.get_robot_response(): {generate_request = }")
        response_count = 1
        if response_count == 1:
            responses = requests.post(endpoint, json=generate_request.to_dict())
            # If bad response
            if responses.status_code != 200:
                logging.error(f"{__class__.__name__}.get_robot_response(): Bad response from llama: {responses.status_code}")
                return None
            

            responses = responses.json()
            response = responses[0]
            
    
        else:
            responses = []
            for _ in range(response_count):
                response = requests.post(endpoint, data=generate_request.to_dict())
                responses.append(response.json()["outputs"][0])

            # Score each output
            output_scores = np.zeros(len(responses))
            longest_output_count = 0
            longest_output_index = 0
            for index in range(len(responses)):
                output = responses[index]
                if len(output) > longest_output_count:
                    longest_output_count = len(output)
                    longest_output_index = index
                sentiment = self.sentiment.get_sentiment(output)
                #response_comment = Comment(self.robot.name, output, sentiment)
                score = sentiment.positive_score
                output_scores[index] = score
                logging.info(f"{__class__.__name__}.{__name__}(): output[{index}] ")
                logging.info(f"{__class__.__name__}.{__name__}(): \t {len(output) = }")
                logging.info(f"{__class__.__name__}.{__name__}(): \t sentiment = {Color.F_Green}{int(100*round(sentiment.positive_score,2))} {Color.F_Red}{int(100*round(sentiment.negative_score,2))} {Color.F_White}")
                logging.info(f"{__class__.__name__}.{__name__}(): \t {output}")
                #logging.info(f"{__class__.__name__}.{__name__}(): \t {comment.printf()}")
            # Get the top scoring index
            top_index = np.argmax(output_scores)
            #logging.info(f"{__class__.__name__}.{func_name}(): Top comment [{top_index}]")
            # Pick the response to use
            response = responses[top_index]

        # Get response
        #response = response.json()
        #generated_text = response["generated_text"]
        #return [generated_text]
        #outputs = response["outputs"]
        #logging.info(f"{__class__.__name__}.get_robot_response(): {len(outputs) = }")

        response = json.loads(response[0])
        logging.info(f"{__class__.__name__}.get_robot_response(): Response generated")
        logging.info(f"{__class__.__name__}.get_robot_response(): {type(response) = }")
        logging.info(f"{__class__.__class__}.get_robot_response(): {response = }")

        # Pull response text
        #response = response["choices"][0]["message"]["content"]

        # Remove prompt
        #response = response.replace(prompt, "")

        endtime = time.time()
        generation_time = endtime - start_time
        self.stats["response_times"].append(generation_time)
        self.stats["tokens_per_sec"] = len(response) / generation_time
        logging.info(f"{__class__.__name__}.get_robot_response(): {generation_time = }")
        logging.info(f"{__class__.__name__}.get_robot_response(): {self.stats['tokens_per_sec'] = }")

        logging.info(f"{__class__.__name__}.get_robot_response(): Response {response = }")
        return response
    
        """
        message_start_token = "<|im_start|>"
        message_end_token = "<|im_end|>"

        # If has the end token
        if message_end_token in response:
            # If has multiple end tokens
            #if response.count(message_end_token) > 1:
                # Get the first instance of the end token
                #first_response = response.split(message_end_token)[0]
                #response = first_response
            # Get the first instance of the end token
            first_response = response.split(message_end_token)[0]
            logging.debug(f"{__class__.__name__}.get_robot_response(): Removing extra dialogue after end token")
            response = first_response
        """
        #if message_start_token in response:
        #    logging.debug(f"{__class__.__name__}.get_robot_response():  Removing extra dialogue after start token")
        #    response = response.split(message_start_token)[1]

        # Show response
        #stop_word = "<|im_end|>"
        #if stop_word in response:
        #    logging.debug(f"{__class__.__name__}.get_robot_response(): Cutting response at {stop_word}")
        #    #response = response[0:response.index(stop_word)]
        #response = response[0:response.index(stop_word)]
        #for stop_word in stopping_words:
        #    if stop_word in response:
        #        logging.debug(f"{__class__.__name__}.get_robot_response(): Cutting response at {stop_word}")
        #        end_index = response.index(stop_word)
        #        logging.debug(f"{__class__.__name__}.get_robot_response(): {end_index = }")
        #        #response = response[0:end_index]
        #        break

        # Show response
        #logging.info(f"{__class__.__name__}.get_robot_response(): Filtered response {response = }")
        #return response

    