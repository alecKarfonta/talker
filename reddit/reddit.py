from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import numpy as np
import pandas as pd

# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument("--headless")  # Run in headless mode
#chrome_options.add_argument("--no-sandbox")
#chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("log-level=3")
chrome_options.add_argument("--lang=en")

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

driver.get("https://www.reddit.com/r/popular/")



# Open the Reddit popular page
# Scroll the page and load more content
def scroll_page(driver, scroll_duration=5, pause_time=2):
    last_height = driver.execute_script("return document.body.scrollHeight")
    end_time = time.time() + scroll_duration
    
    while time.time() < end_time:
        # Scroll down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Wait to load the page
        time.sleep(pause_time)
        
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# Optionally, you can print the page title to confirm it's loaded
print("Page title is:", driver.title)
# Define a function to extract the required information from each post element
def extract_post_info(post_element):
    try:
        title = post_element.get_attribute("post-title")
        author = post_element.get_attribute("author")
        subreddit = post_element.get_attribute("subreddit-prefixed-name")
        post_url = post_element.get_attribute("content-href")
        comment_count = post_element.get_attribute("comment-count")
        score = post_element.get_attribute("score")
        created_timestamp = post_element.get_attribute("created-timestamp")
        permalink = post_element.get_attribute("permalink")
        
        
        return {
            "title": title,
            "author": author,
            "subreddit": subreddit,
            "post_url": post_url,
            "permalink": f"https://www.reddit.com/r/{permalink}",
            "comment_count": comment_count,
            "score": score,
            "created_timestamp": created_timestamp
        }
    except Exception as e:
        print(f"Error extracting post info: {e}")
        return None

def scroll_and_pull():
    # Scroll the page to load more content
    scroll_page(driver)

    print ("Page Scrolled")

    # Find all the post elements on the page
    post_elements = driver.find_elements(By.TAG_NAME, "shreddit-post")

    # Extract information from each post element
    posts_info = []
    for post_element in post_elements:
        post_info = extract_post_info(post_element)
        if post_info:
            posts_info.append(post_info)
            if post_info["title"] not in unique_posts:
                unique_posts[post_info["title"]] = post_info

    print (f"{len(posts_info) =}")

unique_posts = {}

for index in range(10):
    scroll_and_pull()
    print (f"{len(list(unique_posts.keys())) =}")


# Save the unique posts to a CSV file
df = pd.DataFrame(list(unique_posts.values()))
df.to_csv("reddit_popular_posts.csv", index=False)

# Close the driver
driver.quit()