import logging
import time
import json
from openai import OpenAI
import tiktoken
import os
import html



class CommentAnalyzer:
    """
    A class to analyze Reddit comments using OpenAI's GPT model.
    """

    def __init__(self, openai_access_key, log_level=logging.DEBUG):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.model_max_length = 4096
        self.openai_client = OpenAI(api_key=openai_access_key)
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.logger.debug(f"{self.__class__.__name__}: Initialized CommentAnalyzer")

    def call_chatgpt_api(self, prompt, model="gpt-3.5-turbo-instruct", max_tokens=512, temperature=0.7, num_responses=1):
        """
        Make a call to the ChatGPT API.

        :param prompt: The input prompt for the API
        :param model: The GPT model to use (default: "gpt-3.5-turbo-instruct")
        :param max_tokens: Maximum number of tokens in the response (default: 512)
        :param temperature: Sampling temperature (default: 0.7)
        :param responses: Number of responses to generate (default: 1)
        :return: API response
        """
        responses = []
        self.logger.debug(f"{self.__class__.__name__}.call_chatgpt_api(): Calling {model = }]")
        try:
            if model == "gpt-3.5-turbo-instruct":
                response = self.openai_client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    n=num_responses,
                    max_tokens=max_tokens
                )
                for choice in response.choices:
                    responses.append(choice.text)
            elif model == "local":
                host = "threadripper"
                port = 8400
                endpoint = f"http://{host}:{port}/generate_text"
                payload = {
                    "user_prompt": prompt,
                    "echo" : False,
                    "max_tokens": 4096,
                    "response_count": num_responses
                }
                response = requests.post(endpoint, json=payload)
                responses.append(response.json()["generated_text"])
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    n=num_responses,
                    stop=None,
                    temperature=temperature
                )
                for choice in response.choices:
                    responses.append(choice.message.content)    
                
            self.logger.debug(f"{self.__class__.__name__}.call_chatgpt_api(): ChatGPT API call successful")
            return responses
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.call_chatgpt_api(): Error calling ChatGPT API: {str(e)}")
            raise


    def generate_summary(self, post_title, top_comments, max_summary_length=1024):
        try:
            self.logger.debug(f"{self.__class__.__name__}.generate_summary()")
            prompt = f"Summarize and highlight specific examples from the following \n Post: {post_title} \n Comments: {top_comments}"
            prompt_enc = self.enc.encode(prompt)
            token_count = len(prompt_enc)
            self.logger.debug(f"{self.__class__.__name__}.generate_summary(): Token count: {token_count}")

            if token_count > max_summary_length:
                self.logger.warning(f"{self.__class__.__name__}.generate_summary(): Token count exceeds max_summary_length  {max_summary_length}. Truncating prompt")
                prompt = f"Summarize and highlight specific examples from the following \n Post: {post_title} \n Comments: {top_comments[0:3]}"
                prompt_enc = self.enc.encode(prompt)
                token_count = len(prompt_enc)
                self.logger.debug(f"{self.__class__.__name__}.generate_summary(): Token count: {token_count}")

            response = self.call_chatgpt_api(prompt=prompt, max_tokens=max_summary_length)
            summary = response[0]
            return summary
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.generate_summary(): Error generating summary: {str(e)}")
            raise


    def find_max_tokens(self, prompt, requested_token_count):
        try:
            self.logger.debug(f"{self.__class__.__name__}.find_max_tokens()")
            prompt_enc = self.enc.encode(prompt)
            prompt_token_count = len(prompt_enc)
            self.logger.debug(f"{self.__class__.__name__}.find_max_tokens(): Token count: {prompt_token_count}")

            if prompt_token_count + requested_token_count > self.model_max_length:
                self.logger.error(f"{self.__class__.__name__}.find_max_tokens(): Token count exceeds max_model_tokens = {prompt_token_count + requested_token_count}  > {self.model_max_length}. ")
                return None
            elif prompt_token_count > requested_token_count:
                requested_token_count = prompt_token_count + requested_token_count
                self.logger.debug(f"{self.__class__.__name__}.find_max_tokens(): New requested token count: {requested_token_count}")

            return requested_token_count
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.find_max_tokens(): Error finding max tokens: {str(e)}")
            raise


    def generate_topics(self, summary, max_topics_length=1024):
        try:
            self.logger.debug(f"{self.__class__.__name__}.generate_topics()")
            prompt = f"Based on the following dialogue summary, provide several related funny, interesting or deep topics for conversation: {summary}"
            max_len = self.find_max_tokens(prompt, requested_token_count=max_topics_length)

            self.logger.debug(f"{self.__class__.__name__}.generate_topics(): {max_topics_length = }")


            topics = self.call_chatgpt_api(prompt, max_tokens=max_len)
            return topics[0]
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.generate_topics(): Error generating topics: {str(e)}")
            raise


    def generate_responses(self, top_comments, summary, topics, max_responses_length=1024, num_responses=25):
        try:
            self.logger.debug(f"{self.__class__.__name__}.generate_responses()")
            prompt = f"{summary} {topics} Based on the Dialogue Summary and Topics of conversation, pick a topic and write a short funny comment on the subject: "
            max_len = self.find_max_tokens(prompt, requested_token_count=max_responses_length)
            responses = self.call_chatgpt_api(prompt, max_tokens=max_len, num_responses=num_responses)
            possible_responses = [response for response in responses]
            return possible_responses
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.generate_responses(): Error generating responses: {str(e)}")
            raise


    def select_best_response(self, possible_responses, max_response_length=1024):
        try:
            self.logger.debug(f"{self.__class__.__name__}.Select best response()")
            prompt = f"Given the following list of comments select the most funny and clever comment. Refine the comment and reply with the better version. \n {possible_responses}"
            max_len = self.find_max_tokens(prompt, requested_token_count=max_response_length)
            best_response = self.call_chatgpt_api(prompt, max_tokens=max_len)
            return best_response[0]
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Error selecting best response: {str(e)}")
            raise

    def analyze_comments(self, post_title, top_comments, max_summary_length=1024, max_topics_length=1024, max_responses_length=1024):
        try:
            summary = self.generate_summary(post_title, top_comments, max_summary_length)
            time.sleep(5)
            self.logger.debug(f"{self.__class__.__name__}: {len(summary) = }")

            topics = self.generate_topics(summary, max_topics_length)

            self.logger.debug(f"{self.__class__.__name__}: {len(topics) = }")
            time.sleep(5)

            possible_responses = self.generate_responses(top_comments, summary, topics, max_responses_length)
            time.sleep(5)

            self.logger.debug(f"{self.__class__.__name__}: {len(possible_responses) =}")

            best_response = self.select_best_response(possible_responses, max_responses_length)

            return {
                "dialogue_summary": summary,
                "topics": topics,
                "possible_responses": possible_responses,
                "best_response": best_response
            }
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}: Error during comment analysis: {str(e)}")
            raise


    def generate_html_output(self, post_title, top_comments, result):
        # Split the topics string into a list
        topics_list = result["topics"].split('\n')
        # Filter out any empty strings and strip whitespace
        topics_list = [topic.strip() for topic in topics_list if topic.strip()]

        html_output = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reddit Post Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .section {{ margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
                .comment, .response {{ margin-bottom: 10px; }}
                ol {{ padding-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>Reddit Post Analysis</h1>
            
            <div class="section">
                <h2>Post Title</h2>
                <p>{html.escape(post_title)}</p>
            </div>

            <div class="section">
                <h2>Top Comments</h2>
                {"".join(f'<div class="comment"><strong>Comment {i+1}:</strong> {html.escape(comment)}</div>' for i, comment in enumerate(top_comments))}
            </div>

            <div class="section">
                <h2>Dialogue Summary</h2>
                <p>{html.escape(result["dialogue_summary"])}</p>
            </div>

            <div class="section">
                <h2>Topics for Conversation</h2>
                <ol>
                    {"".join(f'<li>{html.escape(topic)}</li>' for topic in topics_list)}
                </ol>
            </div>

            <div class="section">
                <h2>Possible Responses</h2>
                {"".join(f'<div class="response"><strong>Response {i+1}:</strong> {html.escape(response)}</div>' for i, response in enumerate(result["possible_responses"]))}
            </div>

            <div class="section">
                <h2>Best Response</h2>
                <p>{html.escape(result["best_response"])}</p>
            </div>
        </body>
        </html>
        """
        return html_output


# Usage example:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    # Load the comments file
    with open("../reddit_comments.json", "r") as file:
        reddit_comment = json.load(file)

    unique_comments = reddit_comment.get("unique_comments", [])
    top_comments = reddit_comment.get("top_comments", [])
    post_url = reddit_comment.get("url", "")
    post_title = reddit_comment.get("post_title", "")

    print (f"{post_url = }")
    print (f"{len(unique_comments) = }")
    #print (f"{unique_comments = }")
    print (f"{len(top_comments) = }")
    #print (f"{top_comments = }")


    analyzer = CommentAnalyzer(openai_access_key=OPENAI_ACCESS_KEY)
    result = analyzer.analyze_comments(post_title, top_comments)

    html_string = analyzer.generate_html_output(post_title, top_comments, result)   

    html_string = html_string.replace("\n", "")
    html_string = html_string.replace('\"', "")
    if html_string[0] == '"':
        html_string = html_string[1:]


    output_string = f"Reading post {post_title}\n"
    output_string += "-"*100 + "\n"
    output_string += "Top Comments:\n"
    for i, comment in enumerate(top_comments):
        output_string += f"Comment {i+1}: {comment}\n"
    output_string += "-"*100 + "\n"

    output_string += "Dialogue Summary:\n"
    output_string += result["dialogue_summary"] + "\n"
    output_string += "-"*100 + "\n"

    output_string += "Topics for Conversation:\n"
    output_string += result["topics"] + "\n"
    output_string += "-"*100 + "\n"

    output_string += "Possible Responses:\n"
    for i, response in enumerate(result["possible_responses"]):
        output_string += f"Response {i+1}: {response}\n"
    output_string += "-"*100 + "\n"

    output_string += "Best Response:\n"
    output_string += result["best_response"] + "\n"
    output_string += "-"*100 + "\n"

    print(output_string)



    with open("reddit_comment_summary.txt", "w") as file:
        json.dump(output_string, file)

    with open("reddit_comment_summary.html", "w") as file:
        json.dump(html, file)
