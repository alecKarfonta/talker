import os
import time
import pickle
import logging
import googlesearch
import wikipedia

from color import Color

class Info():
    def __init__(self):
        """
        A module that helps to pull information from the internet
        """

        if not os.path.isfile("info_cache.p"):
            logging.info(f"{__class__.__name__}.{__name__}(): Init info cache")
            self.cache = {}
            cache_file = open("info_cache.p", "wb")
            pickle.dump(self.cache, cache_file)
            cache_file.close()
        else:
            cache_file = open("info_cache.p", "rb")
            self.cache = pickle.load(cache_file)
            cache_file.close()
        pass

    def save_cache(self):
        cache_file = open("info_cache.p", "wb")
        pickle.dump(self.cache, cache_file)
        cache_file.close()
    
    def find_wiki_page(self, search_term, max_response_size=64, is_time=True):
        if is_time:
            start_time = time.time()
        logging.info(f"{__class__.__name__}.find_wiki_page(search_term = {search_term = },  {max_response_size = }, {is_time = })")

        # If term exists in cache
       # if search_term in self.cache:
       #     return self.cache[search_term]

        # Search wiki for page
        results = wikipedia.search(search_term)
    
        if not results:
            logging.warning(f"{__class__.__name__}.find_wiki_page(): No results found for {search_term}")
            return None
        
        # Find a response
        response = None
        for index in range(len(results)):
            try:
                # Get the summary
                response = wikipedia.summary(results[index])
                break
            except:
                pass
        if not response:
            logging.error(f"{Color.F_Red}{__class__.__name__}.find_wiki_page(): Could not parse results for {search_term} given results = {results}{Color.F_White}")

        logging.info(f"{__class__.__name__}.find_wiki_page(): {len(response.split(' ')) = }")
        if len(response.split(" ")) > max_response_size:
            logging.info(f"{__class__.__name__}.find_wiki_page(): Reducing size of long response")
            response = response.split(" ")[0:max_response_size]
            response = " ".join(response)


        # Save result in cache
        self.cache[search_term] = response
        self.save_cache()
        
        if is_time:
            runtime = time.time() - start_time
            logging.warning (f"{__class__.__name__}.find_wiki_page(): runtime = {runtime}")
        return response
    