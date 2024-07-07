from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# GLOBAL VARIABLES
# my_model_path = "zephyr-7b-beta.Q4_0.gguf"
my_model_path = "mistral-7b-v0.1-layla-v4-chatml-Q5_K.gguf"
CONTEXT_SIZE = 7999

# LOAD THE MODEL
model = Llama(
            model_path=my_model_path, 
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=64, 
            flash_attn=False,
            n_batch = 512, 
            n_predict = -1, 
            n_keep = 1
            )

logger.info(f"Starting server with model: {my_model_path =}")

# FastAPI app
app = FastAPI()

class PromptRequest(BaseModel):
    user_prompt: str
    max_tokens: int = 100
    temperature: float = 0.3
    top_p: float = 0.1
    echo: bool = True
    overlap: int = 500
    max_attempts: int = 10
    response_count: int = 1
    stop: str = None
    is_repeating: bool = False
    is_story_mode: bool = False
    repeat_penalty: float = 1.2
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

prompt_request = PromptRequest(
    user_prompt="The robot walked into the room and said",
    max_tokens=100,
    temperature=0.3,
    top_p=0.1,
    echo=True,
    overlap=500,
    max_attempts=0,
    stop="None"
)

class TextResponse(BaseModel):
    generated_text: str
    finish_reason: str
    token_count: int




@app.post(
        "/conversation", 
        response_model=TextResponse, 
        summary="Generate Conversation Text", 
        description="Generates text based on the user prompt using the Llama model.")
def conversation_post(request: PromptRequest):
    logger.info(f"{__name__}()")
    model_params = {
        "top_k" : 40, # Default 40
        "top_p" : request.top_p, # Default 0.1
        "temperature" : request.temperature, # Default 0.3
        "max_tokens" : request.max_tokens, # Default 100
        "repeat_penalty" : request.repeat_penalty, # Default 1.1
        "frequency_penalty" : request.frequency_penalty, # Default 0.0
        "presence_penalty" : request.presence_penalty, # Default 0.0
    }
    is_repeating = request.is_repeating
    is_story_mode = request.is_story_mode

    logger.debug(f'{__name__}(): {request.user_prompt = }')
    try:
        model_output = model(
            request.user_prompt,
            #max_tokens=request.max_tokens,
            #temperature=request.temperature,
            #top_p=request.top_p,
            echo=request.echo,
            **model_params
            #stop=request.stop,
        )

        overlap = request.overlap
        max_attempts = request.max_attempts

        logger.debug(f'{__name__}(): {model_output["choices"][0]["text"] = }')

        retries = 0
        prev_prompt = ""
        prev_new_text = ""
        while is_story_mode and \
            model_output["usage"]["completion_tokens"] < request.max_tokens and \
            retries < max_attempts:
            retries += 1

            # Increase temperature
            model_params["temperature"] = request.temperature
            logging.debug(f"-"*100)
            logging.debug(f"Creating more output {retries =}/{max_attempts}")
            logging.debug(f"Output too short {model_output['usage']['completion_tokens'] =}/{request.max_tokens}")
            new_prompt = model_output["choices"][0]["text"][-overlap:]

            # If says The End
            if "The End" in model_output["choices"][0]["text"]:
                model_output["choices"][0]["text"] = model_output["choices"][0]["text"].replace("The End", "")
                # Remove the last few sentences from the prompt
                new_prompt = model_output["choices"][0]["text"][-overlap:]
            
            # Check for repeated prompts
            if prev_prompt == new_prompt \
                or prev_prompt[-50:] == new_prompt[-50:]:
                logging.error(f"{__name__}(): Repeated prompt {new_prompt =}")
                break

            new_prompt = f"Continue {request.user_prompt} Don't repeat sections. Pick up after this snippet: " + new_prompt

            logging.debug(f'Sending new prompt {new_prompt = }')
            logging.debug(f'With params {model_params = }')

            # Get more model output
            more_model_output = model(
                new_prompt,
                #max_tokens=request.max_tokens - model_output["usage"]["completion_tokens"],
                #temperature=request.temperature + (0.03 * np.random.random()),
                #top_p=request.top_p,
                echo=False,
                **model_params
            )
            new_text = more_model_output["choices"][0]["text"]
            logger.debug(f"{__name__}(): {new_text =}")

            # If new text is a duplicate
            if new_text == prev_new_text:
                logger.error(f"generate_text: Repeated new text {new_text =}")

            # If did not produce new text
            if is_story_mode and \
                (new_text == prev_new_text \
                or new_text[-50:] == prev_new_text[-50:] \
                or len(new_text) < 10 \
                or new_text == "The End"):

                logging.warn(f"{__name__}(): Generated short new text {len(new_text) =}")

                # Increase temperature
                model_params["temperature"] = 0.5

                # Start a new chapter
                if is_story_mode:
                    model_output["choices"][0]["text"] += " \n\n Chapter "
                
                # Create new prompt from the last few sentences
                new_prompt = model_output["choices"][0]["text"][-overlap:]
                # Run inference on the new prompt
                more_model_output = model(
                    new_prompt,
                    #max_tokens=request.max_tokens - model_output["usage"]["completion_tokens"],
                    #temperature=request.temperature + (0.03 * np.random.random()),
                    #top_p=request.top_p,
                    echo=False,
                    **model_params
                    #stop=request.stop
                )
                new_text = more_model_output["choices"][0]["text"]
                logger.debug(f"{__name__}(): {new_text =}")

            if new_text == prev_new_text or new_text[-50:] == prev_new_text[-50:]:
                logger.error(f"{__name__}(): Could not resolve repeated new text {new_text =}")
                break


            #combined_text = model_output["choices"][0]["text"].replace(new_prompt, new_text)
            #logger.debug(f"generate_text: {combined_text =}")
            # Append the new output to the previous output
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            model_output["choices"][0]["text"] += new_text
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            model_output["usage"]["completion_tokens"] += more_model_output["usage"]["completion_tokens"]

            logging.debug(f"New length {model_output['usage']['completion_tokens'] =}/{request.max_tokens}")
            logging.debug(f"-"*100)
            prev_prompt = new_prompt
            prev_new_text = new_text
            #if model_output["choices"][0]["finish_reason"] == "stop":
            #    break
        #return_object = TextResponse(
        #    generated_text=model_output["choices"][0]["text"],
        #    finish_reason=model_output["choices"][0]["finish_reason"],
        #    token_count=model_output["usage"]["completion_tokens"]
        #)
        return {
            "generated_text" : model_output["choices"][0]["text"],
            "finish_reason" : model_output["choices"][0]["finish_reason"],
            "token_count" : model_output["usage"]["completion_tokens"]
        }
    except Exception as e:
        logger.error(f"{__name__}(): {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post(
        "/generate_text", 
        response_model=TextResponse, 
        summary="Generate Text", 
        description="Generates text based on the user prompt using the Llama model.")
def generate_text_post(request: PromptRequest):
    logger.info(f"{__name__}()")
    model_params = {
        "top_k" : 40, # Default 40
        "top_p" : request.top_p, # Default 0.1
        "temperature" : request.temperature, # Default 0.3
        "max_tokens" : request.max_tokens, # Default 100
        "repeat_penalty" : 1.2, # Default 1.1
        "frequency_penalty" : 0.0, # Default 0.0
        "presence_penalty" : 0.0, # Default 0.0
    }
    is_repeating = False
    is_story_mode = request.is_story_mode
    try:
        model_output = model(
            request.user_prompt,
            #max_tokens=request.max_tokens,
            #temperature=request.temperature,
            #top_p=request.top_p,
            echo=request.echo,
            **model_params
            #stop=request.stop,
        )

        overlap = request.overlap
        max_attempts = request.max_attempts

        logger.debug(f'{__name__}(): {model_output["choices"][0]["text"] = }')

        retries = 0
        prev_prompt = ""
        prev_new_text = ""
        while is_story_mode and \
            model_output["usage"]["completion_tokens"] < request.max_tokens and \
            retries < max_attempts:
            retries += 1

            # Increase temperature
            model_params["temperature"] = request.temperature
            logging.debug(f"-"*100)
            logging.debug(f"Creating more output {retries =}/{max_attempts}")
            logging.debug(f"Output too short {model_output['usage']['completion_tokens'] =}/{request.max_tokens}")
            new_prompt = model_output["choices"][0]["text"][-overlap:]

            # If says The End
            if "The End" in model_output["choices"][0]["text"]:
                model_output["choices"][0]["text"] = model_output["choices"][0]["text"].replace("The End", "")
                # Remove the last few sentences from the prompt
                new_prompt = model_output["choices"][0]["text"][-overlap:]
            
            # Check for repeated prompts
            if prev_prompt == new_prompt \
                or prev_prompt[-50:] == new_prompt[-50:]:
                logging.error(f"{__name__}(): Repeated prompt {new_prompt =}")
                break

            new_prompt = f"Continue {request.user_prompt} Don't repeat sections. Pick up after this snippet: " + new_prompt

            logging.debug(f'Sending new prompt {new_prompt = }')
            logging.debug(f'With params {model_params = }')

            # Get more model output
            more_model_output = model(
                new_prompt,
                #max_tokens=request.max_tokens - model_output["usage"]["completion_tokens"],
                #temperature=request.temperature + (0.03 * np.random.random()),
                #top_p=request.top_p,
                echo=False,
                **model_params
            )
            new_text = more_model_output["choices"][0]["text"]
            logger.debug(f"{__name__}(): {new_text =}")

            # If new text is a duplicate
            if new_text == prev_new_text:
                logger.error(f"generate_text: Repeated new text {new_text =}")

            # If did not produce new text
            if is_story_mode and \
                (new_text == prev_new_text \
                or new_text[-50:] == prev_new_text[-50:] \
                or len(new_text) < 10 \
                or new_text == "The End"):

                logging.warn(f"{__name__}(): Generated short new text {len(new_text) =}")

                # Increase temperature
                model_params["temperature"] = 0.5

                # Start a new chapter
                if is_story_mode:
                    model_output["choices"][0]["text"] += " \n\n Chapter "
                
                # Create new prompt from the last few sentences
                new_prompt = model_output["choices"][0]["text"][-overlap:]
                # Run inference on the new prompt
                more_model_output = model(
                    new_prompt,
                    #max_tokens=request.max_tokens - model_output["usage"]["completion_tokens"],
                    #temperature=request.temperature + (0.03 * np.random.random()),
                    #top_p=request.top_p,
                    echo=False,
                    **model_params
                    #stop=request.stop
                )
                new_text = more_model_output["choices"][0]["text"]
                logger.debug(f"{__name__}(): {new_text =}")

            if new_text == prev_new_text or new_text[-50:] == prev_new_text[-50:]:
                logger.error(f"{__name__}(): Could not resolve repeated new text {new_text =}")
                break


            #combined_text = model_output["choices"][0]["text"].replace(new_prompt, new_text)
            #logger.debug(f"generate_text: {combined_text =}")
            # Append the new output to the previous output
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            model_output["choices"][0]["text"] += new_text
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            #model_output["choices"][0]["text"] += "\n\n---------------------------------------------------------------------------------------------------\n\n"
            model_output["usage"]["completion_tokens"] += more_model_output["usage"]["completion_tokens"]

            logging.debug(f"New length {model_output['usage']['completion_tokens'] =}/{request.max_tokens}")
            logging.debug(f"-"*100)
            prev_prompt = new_prompt
            prev_new_text = new_text
            #if model_output["choices"][0]["finish_reason"] == "stop":
            #    break
        #return_object = TextResponse(
        #    generated_text=model_output["choices"][0]["text"],
        #    finish_reason=model_output["choices"][0]["finish_reason"],
        #    token_count=model_output["usage"]["completion_tokens"]
        #)
        return {
            "generated_text" : model_output["choices"][0]["text"],
            "finish_reason" : model_output["choices"][0]["finish_reason"],
            "token_count" : model_output["usage"]["completion_tokens"]
        }
    except Exception as e:
        logger.error(f"{__name__}(): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="Health Check", description="Checks the health of the server.")
def health_check():
    return {"status": "OK"}


@app.get("/", summary="Root Endpoint", description="Returns a welcome message.")
def read_root():
    return {"message": "Welcome to the Text Generation API"}
