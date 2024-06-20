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
    stop: str = None


class TextResponse(BaseModel):
    generated_text: str
    finish_reason: str
    token_count: int



@app.post(
        "/generate_text", 
        response_model=TextResponse, 
        summary="Generate Text", 
        description="Generates text based on the user prompt using the Llama model.")
def generate_text(request: PromptRequest):
    logger.info(f"generate_text()")
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

        logger.debug(f'generate_text(): {model_output["choices"][0]["text"] = }')

        retries = 0
        prev_prompt = ""
        prev_new_text = ""
        while model_output["usage"]["completion_tokens"] < request.max_tokens and retries < max_attempts:
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
                logging.error(f"generate_text(): Repeated prompt {new_prompt =}")
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
            logger.debug(f"generate_text: {new_text =}")

            # If new text is a duplicate
            if new_text == prev_new_text:
                logger.error(f"generate_text: Repeated new text {new_text =}")

            # If did not produce new text
            if new_text == prev_new_text \
                or new_text[-50:] == prev_new_text[-50:] \
                or len(new_text) < 10 \
                or new_text == "The End":

                logging.warn(f"Generated short new text {len(new_text) =}")

                # Increase temperature
                model_params["temperature"] = 0.5

                # Start a new chapter
                model_output["choices"][0]["text"] += " \n\n Chapter "
                new_prompt = model_output["choices"][0]["text"][-overlap:]
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
                logger.debug(f"generate_text: {new_text =}")

            if new_text == prev_new_text or new_text[-50:] == prev_new_text[-50:]:
                logger.error(f"generate_text: Could not resolve repeated new text {new_text =}")
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


@app.get("/", summary="Root Endpoint", description="Returns a welcome message.")
def read_root():
    return {"message": "Welcome to the Text Generation API"}
