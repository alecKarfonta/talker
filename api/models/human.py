import os
import pickle
import logging
from color import Color
from comment import Comment

class Memory():
    def __init__(self, prompt:Comment, comment:Comment, response:Comment):
        '''
            Parameters
            ----------
            prompt : Comment 
                The last section of the prompt provided to the robot
            comment : Comment
                The robot generated response to the prompt
            repsonse : Comment
                The user response to the robot comment
        '''
        self.prompt = prompt
        self.comment = comment
        self.response = response
    
    
    def __repr__(self):
        out_str = f"prompt: {self.prompt.printf()}\n"
        out_str += f"comment: {self.comment.printf()}\n"
        out_str += f"response: {self.response.printf()}\n"
        return out_str
        
    
class Human():
    def __init__(self,
                 name:str,
                 use_cache:bool=True
                ):
        
        self.cache_filename = f"{name}_human.p"
        if use_cache and os.path.isfile(self.cache_filename):
            loaded = self.load()
            if not loaded:
                logging.error(f"{Color.F_Red}{__class__.__name__}.__init__(): Failed to load chat history {self.cache_filename}{Color.F_White}")
        
        # Else init fresh
        else:
            self.name = name
            self.positive_memories = []
            self.negative_memories = []
            self.save()
    
    def save(self):
        pickle.dump({"name":self.name, "positive_memories":self.positive_memories, "negative_memories":self.negative_memories}, open(self.cache_filename, "wb"))
        return True

    def load(self):
        if not os.path.isfile(self.cache_filename):
            logging.warning(f"{__class__.__name__}.load(): File does not exist")
            return None
        
        saved = pickle.load(open(self.cache_filename, "rb"))

        self.name = saved["name"]
        self.positive_memories = saved["positive_memories"]
        self.negative_memories = saved["negative_memories"] # list of Comment objects

        return True
    
    def add_positive_memory(self, memory):
        self.positive_memories = [memory] + self.positive_memories
        self.save()
        
    def add_negative_memory(self, memory):
        self.negative_memories = [memory] + self.negative_memories
        self.save()
            
    def __repr__(self):
        return f"Human({self.name})"