import re
import time
import json
import torch
import num2words
import pickle
import logging
from copy import copy
import transformers
from transformers import BitsAndBytesConfig
from transformers import  GenerationConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
#from accelerate import init_empty_weights

from numpy.random import random, choice
#from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
#from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

#from speechbrain.pretrained import Tacotron2
#from speechbrain.pretrained import HIFIGAN

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import sys
sys.path.append("..") 
sys.path.append("../shared/") 

from color import Color

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
                 model_name:str="PygmalionAI/pygmalion-6b",
                 model_file:str=None,
                 finetune_path:str=None,
                 is_debug=False,
                 is_use_gpu=True,
                 is_use_bnb=True
                ):
        logging.info(f"{__class__.__name__}.{__name__}(): (name={name}, persona={persona}, model_name={model_name})")
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
        self.model_name = model_name
        self.model_file = model_file
        self.finetune_path = finetune_path
        self.model = None
        self.is_use_bnb = is_use_bnb
        self.is_use_gpu = is_use_gpu
        self.stats = {
            "tokens_per_sec" : 0,
            "response_times" : []
        }
        self.max_generation_time = 10
        self.init_models()
        replace_list.extend([
            (f"{self.name} jor", " "),
            (f"{self.name}: jor", f"{self.name}: "),
            (f"{self.name}: jor jor", f"{self.name}: "),
            (f"Major: jor jor", f"{self.name}: ")
        ]
        )

        #logging.info(f"{__class__.__name__}.{__name__}(): Init voice model")

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
            "model_name" : self.model_name,
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
        self.model_name = values["model_name"]
        self.model_file = values["model_file"]
        self.finetune_path = values["finetune_path"]
        

    def init_models(self):
        # If using bits and bytes: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        if self.is_use_bnb:
            logging.info(f"{__class__.__name__}.{__name__}(): Using bnb to quantize model")
            # Init bits and bytes config
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            if self.model_file:
                logging.info(f"{__class__.__name__}.{__name__}(): Load saved model {self.model_file}")
                self.model_source = self.model_file
                self.model = AutoModelForCausalLM.from_pretrained(self.model_file, local_files_only=True, quantization_config=nf4_config)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
            elif self.finetune_path:
                logging.info(f"{__class__.__name__}.{__name__}(): Load fine tuned model: {self.finetune_path}")
                self.model_source = self.finetune_path
                self.model = AutoModelForCausalLM.from_pretrained(self.finetune_path, quantization_config=nf4_config)
                self.tokenizer = AutoTokenizer.from_pretrained(self.finetune_path)
            elif self.model_name:
                logging.info(f"{__class__.__name__}.{__name__}(): Init mew Model: {self.model_name}")
                self.model_source = self.model_name
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=nf4_config)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                #config = AutoConfig.from_pretrained(model_name)
                ##with init_empty_weights():
                #self.model = AutoModelForCausalLM.from_config(config)
        else:
            if self.model_file:
                logging.info(f"{__class__.__name__}.{__name__}(): Load saved model {self.model_file}")
                self.model_source = self.model_file
                self.model = AutoModelForCausalLM.from_pretrained(self.model_file, local_files_only=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
            elif self.finetune_path:
                logging.info(f"{__class__.__name__}.{__name__}(): Load fine tuned model: {self.finetune_path}")
                self.model_source = self.finetune_path
                self.model = AutoModelForCausalLM.from_pretrained(self.finetune_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.finetune_path)
            elif self.model_name:
                logging.info(f"{__class__.__name__}.{__name__}(): Init mew Model: {self.model_name}")
                self.model_source = self.model_name
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                device_map="cuda")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                #config = AutoConfig.from_pretrained(model_name)
                ##with init_empty_weights():
                #self.model = AutoModelForCausalLM.from_config(config)


        # If not using bits and bytes to control model devive deployment
        if not self.is_use_bnb:
            # Set precision to 16 bit
            logging.info(f"{__class__.__name__}.{__name__}(): Setting precision to fp16")
            #self.model.half()
            # If is using gpu
            if self.is_use_gpu:
                # Send model to gpu
                logging.info(f"{__class__.__name__}.{__name__}(): Send model to gpu")
                self.model.to("cuda")
        logging.info(f"{__class__.__name__}.{__name__}(): Done")

        return True
    

    def get_robot_response(self,
                           person:str,
                           prompt:str,
                           min_len:int=128,
                           max_len:int=256,
                           response_count = 1
                          ):
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
        # Save start time
        start_time = time.time()
        # Check if needs to init models
        if not self.model:
            logging.error(f"{Color.F_Red}{__class__.__name__}.get_robot_response(): Error models not intialized{Color.F_White}")
            # Init models
            self.init_models(self.model_file, self.model_name, self.finetune_path)
        
        # Clear memory
        logging.info(f"{__class__.__name__}.get_robot_response(): Clearing gpu memory")
        torch.cuda.empty_cache()

        # Randomly prepend the output with the person's name
        #if random() > .85:
        #    prompt = prompt + f"Well {person} "

        # If prompt is longer than max allowed input size
        if len(prompt.split(" ")) > 1024:
            logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Prompt too long: {len(prompt.split(' '))}. Truncating {Color.F_White}")
            prompt = prompt.split(" ")
            prompt = prompt[-1024:]
            prompt = " ".join(prompt)

        # Encode input strings
        tokenized_items = self.tokenizer(prompt, return_tensors="pt").to("cuda" if self.is_use_gpu else "cpu")
        prompt_token_count = len(tokenized_items["input_ids"][0])

        # Show input token length
        logging.info(f"{__class__.__name__}.get_robot_response(): input token length = {tokenized_items['input_ids'].shape}")

        # Check if the input is larger that the max allowed input
        if tokenized_items['input_ids'].shape[0] > self.context_token_limit:
            logging.warning(f"{__class__.__name__}.get_robot_response(): Context length too long. tokenized_items['input_ids'].shape[0] = {tokenized_items['input_ids'].shape[0]}")

        # Check that the input and model are on the same device
        if tokenized_items["input_ids"].get_device() != self.model.device.index:
            logging.error(f"{Color.F_Red}{__class__.__name__}.get_robot_response(): input and model on difference devices{Color.F_White}")
            logging.info(f"{__class__.__name__}.get_robot_response(): input device = {tokenized_items['input_ids'].get_device()}")
            logging.info(f"{__class__.__name__}.get_robot_response(): model device = {self.model.device.index}")

        # If needs to generate more than one response
        if response_count > 1:
            # For each response required
            for index in range(response_count-1):
                # Append a copy of the input to itself
                tokenized_items["input_ids"] = torch.cat((tokenized_items["input_ids"], tokenized_items["input_ids"].clone()), dim=0)
                tokenized_items["attention_mask"] = torch.cat((tokenized_items["attention_mask"], tokenized_items["attention_mask"].clone()), dim=0)

        # Create stopping criteria for generation
        stopping_words = copy(self.stopping_words)
        stopping_words.extend([
                            f"\n{person} ", 
                            f"\n{person}:", 
                            f"{person}:", 
                            f"{person.upper()}:",  
        ])
        stopping_list = []
        for stopping_word in stopping_words:
            stopping_list.append(_SentinelTokenStoppingCriteria(
                sentinel_token_ids=self.tokenizer(
                    stopping_word,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to("cuda" if self.is_use_gpu else "cpu"),
                starting_idx=tokenized_items.input_ids.shape[-1]))            
        stopping_criteria_list = StoppingCriteriaList(stopping_list)

        # Show stopping words
        logging.info(f"{__class__.__name__}.get_robot_response(): stopping_words = {stopping_words}")
        
        # Config genration
        generation_config = GenerationConfig(
                                             min_len=min_len+prompt_token_count,
                                             max_len=max_len+prompt_token_count,
                                             min_new_tokens=min_len, 
                                             max_new_tokens=max_len+100, 
                                             do_sample=True, 
                                             top_k=3, # Default = 50
                                             temperature=1.0, # Default = 1.0
                                             #epsilon_cutoff=0.0004,
                                             #length_penalty=5.0, # Default = 1.0,
                                             #num_return_sequences=4, # Default=1.0
                                             #num_beams=4, # Default = 1.0
                                             #num_beam_groups=2, # Default = 1.0
                                             #diversity_penalty=5.0, # Default = 0.0,
                                             #repetition_penalty=1.0, # Default = 1.0
                                             #encoder_repetition_penalty=1.15, # Default 1.0
                                             #bad_words_ids=self.tokenizer(stopping_words).input_ids,
                                             #force_words_ids=self.tokenizer(person).input_ids,
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             bos_token_id=self.tokenizer.bos_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             max_time=8,
                                             
        )

        logging.info(f"{__class__.__name__}.get_robot_response(): Generating output")
        # Generate output logits from model
        with torch.no_grad():
            logits = self.model.generate(
                                        **tokenized_items,
                                        #stopping_criteria=stopping_criteria_list, 
                                        generation_config=generation_config,
                                        )
        # Show output logits shape
        logging.info(f"{__class__.__name__}.get_robot_response(): {logits.shape = }")
        #logging.info(f"{__class__.__name__}.get_robot_response(): {logits = }")

        #stopping_tokens = self.tokenizer(self.stopping_words).input_ids
        #for index in range(len(logits)):
        #    for stopping_token in stopping_tokens:
        #        if stopping_token in logits[index]:
        #            logging.info(f"{__class__.__name__}.get_robot_response(): Encountered stopping word{self.tokenizer.decode(stopping_token)}")
                    

        # Decode outputs
        unprocessed_outputs = self.tokenizer.batch_decode(logits, skip_special_tokens=False)
        # Save a list of all possible outputs generated
        processed_outputs = []
        # For each output
        for output_index in range(len(unprocessed_outputs)):
            # Decode output logits to words
            #output = self.tokenizer.decode(logits[output_index], skip_special_tokens=True)
            output = unprocessed_outputs[output_index]
            # Filter input
            output = output[len(prompt):]
            # Show unprocessed output
            logging.info(f"{__class__.__name__}.get_robot_response(): Unprocessed output = {output}")
            # Filter own name prompts
            output = output.replace("lly:", "")
            output = output.replace("Bi ", "")

            # Filter text
            for item in replace_list:
                output = output.replace(item[0], item[1])
            for item in remove_list:
                output = output.replace(item,"")
            #logging.info(f"{__class__.__name__}.get_robot_response(): Before output processing = {output}")
            # Count tokens in output
            token_count = len(output.split(" "))
            # Show count of tokens generated
            logging.info(f"{__class__.__name__}.get_robot_response(): {token_count = }")
            # Check if made no tokens
            if token_count == 0:
                logging.error(f"{__class__.__name__}.get_robot_response(): No output tokens generated")
                continue

            # Check if got enough output
            if token_count < 2:
                logging.warning(f"{__class__.__name__}.get_robot_response(): output token length too short : len(output) = {len(output)}")
                continue
            
            # Check if output is too long
            if token_count > 2000:
                logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): output token length too long : len(output) = {len(output)}{Color.F_White}")
                logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): truncating to 2000{Color.F_White}")
                output = output[0:2000]
                #continue

            #print (f"get_robot_response(): Processing output")
            for stop_word in stopping_words:
                if stop_word in output:
                    logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Foudn stopping word = {stop_word} {Color.F_White}")
                    output = output[0:output.index(stop_word)]
            
            # Remove whitespace
            output = output.rstrip()
            
            # Replace user token with actual name
            if "<USER>" in output:
                output = output.replace("<USER>", person)

            # Remove visual expressions
            # TODO: save these to feed into animations
            run_count = 0
            max_run_count = 10
            # Use regex to match strings like *[TEXT]* 
            #match = re.search("(?<=\*)(.*?)(?=\*)", output)
            #while match and run_count < max_run_count:
            #    run_count += 1
            #    #logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Removing [TEXT] from {match.start()-1} to {match.end()+1} {Color.F_White}")
            #    output = f"{output[0:match.start()-1]}{output[match.end()+1:]}"
            #    match = re.search("(?<=\*)(.*?)(?=\*)", output)
            #run_count = 0
            # Use regex to match strings like [[TEXT]] 
            #match = re.search("(?<=\[)(.*?)(?=\])", output)
            #while match and run_count < max_run_count:
            #    run_count += 1
            #    #logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Removing [TEXT] from {match.start()-1} to {match.end()+1} {output[match.start:match.end]} {Color.F_White}")
            #    output = f"{output[0:match.start()-1]}{output[match.end()+1:]}"
            #    match = re.search("(?<=\*)(.*?)(?=\*)", output)
                
            # Strip whitepsace
            output = output.strip()
            
            # Remove all added prompts
            for spice in self.prompt_spices:
                output = output.replace(spice, "")
            for emotion in self.prompt_emotions:
                output = output.replace(f"Be {emotion}.", "")
                
            if len(output) > max_len * 2:
                logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): Output too large len(output) = {len(output)}{Color.F_White}")
                #output = output[0:max_len]
                
            # If bad word in 
            for index in range(len(self.filter_list)):
                filter_words = self.filter_list[index][0]
                replace_words = self.filter_list[index][1]
                for filter_word in filter_words:
                    if filter_word in output:
                        replacement = choice(replace_words)
                        logging.info(f"{__class__.__name__}.get_robot_response(): Replacing {filter_word}, {replacement}")
                        output = output.replace(filter_word, replacement)
            
            output = output.replace("Hehe", "Haha")
            output = output.replace("hehe", "haha")
            # Replace numbers with words for the speach-to-text
            output = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), output)
            #output = output.replace(f"Well {person} Well Alec", "haha")
            #logging.info(f"{__class__.__name__}.get_robot_response(): After output processing = {output}")

            # Check if after processing there is nothing left
            if len(output) == 0:
                logging.warning(f"{Color.F_Yellow}{__class__.__name__}.get_robot_response(): No output after processing: {unprocessed_outputs[output_index]}")


            # Save processing output
            processed_outputs.append(output)

            # Clear memory
            logging.info(f"{__class__.__name__}.get_robot_response(): Clearing gpu memory")
            torch.cuda.empty_cache()
        

        # Check if got any outputs after processing
        if len(processed_outputs) == 0:
            logging.warning(f"{__class__.__name__}.get_robot_response(): Generated no valid outputs")
        
        #print (f"get_robot_response(): Done")
        runtime = time.time() - start_time
        total_tokens = 0
        for index in range(len(processed_outputs)):
            total_tokens += len(processed_outputs[index].split(" "))
        tokens_per_sec = total_tokens / runtime
        self.stats["tokens_per_sec"] = (0.5 * self.stats["tokens_per_sec"]) + (0.5 * tokens_per_sec)
        self.stats["response_times"].append(runtime)
        logging.info(f"{__class__.__name__}.get_robot_response(): runtime = {runtime}")
        logging.info(f"{__class__.__name__}.get_robot_response(): tokens_per_sec = {tokens_per_sec}")
        logging.info(f"{__class__.__name__}.get_robot_response(): overall tokens_per_sec = {self.stats['tokens_per_sec']}")
        return processed_outputs
    
class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False