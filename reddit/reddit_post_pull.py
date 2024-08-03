import logging
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from openai import OpenAI
import tiktoken
import os



class RedditScraper:
    """
    A class to scrape comments from Reddit posts using Selenium.
    """

    def __init__(self, log_level=logging.DEBUG):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.driver = None
        self.logger.debug(f"{self.__class__.__name__}: Initialized RedditScraper")

    def setup_chrome_driver(self):
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("log-level=3")
            #chrome_options.add_argument("--headless")
            chrome_options.add_argument("--lang=en")
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.logger.debug(f"{self.__class__.__name__}: Chrome driver set up successfully")
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Failed to set up Chrome driver: {str(e)}")
            raise

    def scroll_page(self, scroll_duration=5, pause_time=2):
        try:
            self.logger.debug(f"{self.__class__.__name__}: Starting page scroll")
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            end_time = time.time() + scroll_duration
            
            while time.time() < end_time:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(pause_time)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            self.logger.debug(f"{self.__class__.__name__}: Page scroll completed")
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Error during page scroll: {str(e)}")
            raise

    def extract_text_from_comments(self):
        try:
            self.logger.debug(f"{self.__class__.__name__}: Extracting text from comments")
            comments = self.driver.find_elements(By.TAG_NAME, "shreddit-comment")
            texts_to_extract = []

            for comment in comments:
                comment_content = comment.find_element(By.CSS_SELECTOR, "div[id$='-comment-rtjson-content']")
                comment_text = ""
                paragraphs = comment_content.find_elements(By.TAG_NAME, "p")

                for paragraph in paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        comment_text += text + " "
                texts_to_extract.append(comment_text)

            self.logger.debug(f"{self.__class__.__name__}: Extracted {len(texts_to_extract)} comments")
            return texts_to_extract
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Error extracting text from comments: {str(e)}")
            raise

    def scrape_reddit_post(self, url, scroll_time=2, pause_time=2, wait_time=5, scroll_iteration_count=1, top_comment_count=10):
        try:
            self.logger.info(f"{self.__class__.__name__}: Scraping Reddit post: {url}")
            self.setup_chrome_driver()
            self.driver.get(url)
            time.sleep(5)

            unique_comments = {}

            # Pull the initial page
            texts_to_extract = self.extract_text_from_comments()
            for text in texts_to_extract:
                if text not in unique_comments:
                    unique_comments[text] = {
                        "count": 1,
                        "username" : None,
                        "text" : text
                    }
            # Scroll the page and continue grabbing unique comments
            for index in range(scroll_iteration_count):
                self.scroll_page(scroll_duration=2, pause_time=2)
                texts_to_extract = self.extract_text_from_comments()
                for text in texts_to_extract:
                    if text not in unique_comments:
                        unique_comments[text] = {
                            "count": 1,
                            "username" : None,
                            "text" : text
                        }
            # Extract the top comments
            top_comments = texts_to_extract[:top_comment_count]
            # Filter empty string top comments
            top_comments = [comment for comment in top_comments if comment]

            self.logger.debug(f"{self.__class__.__name__}: Extracted top comments: {len(top_comments) = }")
            self.logger.debug(f"{self.__class__.__name__}: {len(texts_to_extract) = }")

            return unique_comments, top_comments
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Error during Reddit post scraping: {str(e)}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.debug(f"{self.__class__.__name__}: Chrome driver closed")


# Usage example:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    post_url = "https://www.reddit.com/r/nottheonion/comments/1dvdzjb/biden_tells_democratic_governors_he_needs_more/"
    post_title = "Biden tells Democratic governors he needs more sleep and plans to stop scheduling events after 8 p.m."

    scraper = RedditScraper()
    unique_comments, top_comments = scraper.scrape_reddit_post(post_url)

    output = {
        "post_title": post_title,
        "url": post_url,
        "top_comments": top_comments,
        "unique_comments": unique_comments
    }

    # Write to local file
    with open("reddit_comments.json", "w") as file:
        json.dump(output, file)
        file.close()


    # Load the comments file
    with open("reddit_comments.json", "r") as file:
        unique_comments = json.load(file)

    print (f"{len(unique_comments) = }")
    print (f"{unique_comments = }")


    with open("reddit_comment.txt", "w") as file:
        json.dump({"post_title" : post_title, "url": post_url, "top_comments": top_comments, "unique_comments" : unique_comments}, file)
