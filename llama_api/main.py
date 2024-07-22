from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from llama_cpp import Llama
import os
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download, list_repo_files
import time

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
my_model_path = "models/nous-hermes-2-mistral-7B-DPO/OpenHermes-2.5-Mistral-7B-DPO-F16.gguf"
#my_model_path = "models/nous-hermes-2-mistral-7B-DPO/OpenHermes-2.5-Mistral-7B-DPO-Q5_K_M.gguf"

MODELS_DIR = "models"
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


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
    max_tokens: int = Field(default=100, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.1, ge=0.0, le=1.0)
    echo: bool = True
    overlap: int = Field(default=500, ge=0)
    max_attempts: int = Field(default=10, ge=1)
    response_count: int = Field(default=1, ge=1)
    stop: Optional[str] = None
    is_repeating: bool = False
    is_story_mode: bool = False
    repeat_penalty: float = Field(default=1.2, ge=0.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0)
    presence_penalty: float = Field(default=0.0, ge=0.0)

class TextResponse(BaseModel):
    outputs: list
    finish_reasons: list
    token_count: int

class BenchmarkRequest(BaseModel):
    num_tokens: int = Field(default=1000, ge=1, description="Number of tokens to generate")
    prompt: str = Field(default="Once upon a time", description="Prompt to use for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for text generation")
    num_runs: int = Field(default=3, ge=1, le=10, description="Number of benchmark runs")

class BenchmarkResponse(BaseModel):
    model_name: str
    average_tokens_per_second: float
    total_time: float
    num_tokens: int
    num_runs: int
    individual_run_results: list



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
        "response_count" : request.response_count
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
    logger.info(f"generate_text_post()")
    model_params = {
        "top_k" : 40, # Default 40
        "top_p" : request.top_p, # Default 0.1
        "temperature" : request.temperature, # Default 0.3
        "max_tokens" : request.max_tokens, # Default 100
        "repeat_penalty" : 1.2, # Default 1.1
        "frequency_penalty" : 0.0, # Default 0.0
        "presence_penalty" : 0.0, # Default 0.0
        #"response_count" : request.response_count
    }
    logger.info(f"generate_text_post(): {model_params = }")
    if model is None:
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please use /change_model to load a model first.")
    

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

        outputs = []
        finish_reasons = []
        for choice in model_output["choices"]:
            outputs.append(str(choice["text"]))
            finish_reasons.append(str(choice["finish_reason"]))

        return {
            "outputs" : outputs,
            "finish_reasons" : finish_reasons,
            "token_count" : model_output["usage"]["completion_tokens"]
        }
    except Exception as e:
        logger.error(f"{__name__}(): {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/benchmark", response_model=BenchmarkResponse, summary="Benchmark Model Performance", description="Evaluates the tokens generated per second by the model")
def benchmark(request: BenchmarkRequest):
    logger.info(f"Starting benchmark with {request.num_tokens} tokens, {request.num_runs} runs")
    
    individual_run_results = []
    total_time = 0

    try:
        for run in range(request.num_runs):
            start_time = time.time()
            
            model_output = model(
                request.prompt,
                max_tokens=request.num_tokens,
                temperature=request.temperature,
                echo=False
            )
            
            end_time = time.time()
            run_time = end_time - start_time
            tokens_generated = model_output["usage"]["completion_tokens"]
            tokens_per_second = tokens_generated / run_time
            
            individual_run_results.append({
                "run": run + 1,
                "time": run_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second
            })
            
            total_time += run_time
            logger.info(f"Benchmark run {run + 1} completed: {tokens_per_second:.2f} tokens/second")

        average_tokens_per_second = sum(run["tokens_per_second"] for run in individual_run_results) / request.num_runs

        return BenchmarkResponse(
            model_name=my_model_path,
            average_tokens_per_second=average_tokens_per_second,
            total_time=total_time,
            num_tokens=request.num_tokens,
            num_runs=request.num_runs,
            individual_run_results=individual_run_results
        )
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


class ModelDownloadRequest(BaseModel):
    repo_id: str = Field(..., description="Hugging Face model repository ID")
    filenames: Optional[str] = Field(None, description="Specific filename to download (optional)")
    auth_token: Optional[str] = Field(None, description="Hugging Face authentication token")


class ModelDownloadResponse(BaseModel):
    status: str
    message: str
    local_path: Optional[str]



def download_model(repo_id: str, filenames: Optional[List[str]] = None, auth_token: Optional[str] = None):
    try:
        logger.info(f"Starting download of model: {repo_id}")
        local_dir = os.path.join(MODELS_DIR, repo_id.split('/')[-1])
        os.makedirs(local_dir, exist_ok=True)
        
        if not filenames:
            # If no specific files are requested, download all files
            filenames = list_repo_files(repo_id, token=auth_token)
            logging.debug(f"download_model(): Downloading files: {filenames = }")
        
        for filename in filenames:
            try:
                logging.debug(f"download_model(): Downloading file: {filename = }")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    token=auth_token
                )
                logger.info(f"Downloaded: {filename} to {file_path}")
            except Exception as e:
                logger.error(f"Error downloading {filename}: {str(e)}")
        
        logger.info(f"Model files downloaded successfully to: {local_dir}")
        return local_dir
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

@app.post("/download_model", response_model=ModelDownloadResponse, summary="Download Model from Hugging Face", description="Downloads all or specific files of a model from Hugging Face Hub with optional authentication")
async def download_model_endpoint(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    try:
        # Start the download process in the background
        background_tasks.add_task(download_model, request.repo_id, request.filenames, request.auth_token)
        
        return ModelDownloadResponse(
            status="started",
            message=f"Model download started for {request.repo_id}. Check logs for progress.",
            local_path=None
        )
    except Exception as e:
        logger.error(f"Error initiating model download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start model download: {str(e)}")




from typing import Optional, List, Dict
import json

class ModelFile(BaseModel):
    name: str
    size: int  # in bytes
    format: str

class ConfigFile(BaseModel):
    name: str
    content: Dict

class ModelInfo(BaseModel):
    name: str
    files: List[ModelFile]
    config: Optional[ConfigFile]

class ListModelsResponse(BaseModel):
    models: List[ModelInfo]

def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)

def get_file_format(file_name: str) -> str:
    extension = os.path.splitext(file_name)[1].lower()
    if extension in ['.safetensors', '.gguf', '.bin', '.pt', '.pth']:
        return extension[1:]  # Remove the leading dot
    return 'unknown'

def read_config_file(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Error decoding JSON from {file_path}")
        return {}

@app.get("/list_models", response_model=ListModelsResponse, summary="List Downloaded Models", description="Lists all models downloaded through the API with detailed information")
def list_models():
    try:
        model_infos = []
        for model_dir in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_dir)
            if os.path.isdir(model_path):
                files = []
                config = None
                for root, _, filenames in os.walk(model_path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        file_format = get_file_format(filename)
                        if file_format != 'unknown':
                            files.append(ModelFile(
                                name=filename,
                                size=get_file_size(file_path),
                                format=file_format
                            ))
                        elif filename == 'config.json':
                            config = ConfigFile(
                                name=filename,
                                content=read_config_file(file_path)
                            )
                
                model_infos.append(ModelInfo(
                    name=model_dir,
                    files=files,
                    config=config
                ))
        
        return ListModelsResponse(models=model_infos)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")




class ChangeModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to the new model file")
    n_ctx: int = Field(default=CONTEXT_SIZE, description="Context size for the model")
    n_gpu_layers: int = Field(default=64, description="Number of GPU layers to use")
    
class ChangeModelResponse(BaseModel):
    status: str
    message: str


import gc
#import torch

def clear_vram():
    global model
    if model is not None:
        del model
        model = None
    
    gc.collect()
    
    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    #    torch.cuda.ipc_collect()
    
    logger.info("VRAM cleared successfully")

def initialize_model(model_path: str, n_ctx: int, n_gpu_layers: int):
    global model
    clear_vram()  # Clear VRAM before initializing new model
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            flash_attn=False,
            n_batch=512,
            n_predict=-1,
            n_keep=1
        )
        logger.info(f"Model initialized successfully: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

@app.post("/change_model", response_model=ChangeModelResponse, summary="Change Current Model", description="Changes the currently loaded model to a new one specified by the given path")
async def change_model(request: ChangeModelRequest):
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        success = initialize_model(request.model_path, request.n_ctx, request.n_gpu_layers)
        
        if success:
            return ChangeModelResponse(
                status="success",
                message=f"Model changed successfully to: {request.model_path}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize the new model")
    except Exception as e:
        logger.error(f"Error changing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to change model: {str(e)}")


@app.get("/clear_model", response_model=ChangeModelResponse, summary="Clear Current Model", description="Unloads the current model and clears VRAM")
async def clear_current_model():
    try:
        clear_vram()
        return ChangeModelResponse(
            status="success",
            message="Current model unloaded and VRAM cleared successfully"
        )
    except Exception as e:
        logger.error(f"Error clearing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear model: {str(e)}")


# Update the root endpoint to check if a model is loaded
@app.get("/", summary="Root Endpoint", description="Returns a welcome message and current model status")
def read_root():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "Welcome to the Text Generation API with Benchmarking, Comprehensive Model Download, Enhanced Model Listing, and Model Switching capabilities",
        "model_status": model_status
    }




@app.get("/health", summary="Health Check", description="Checks the health of the server.")
def health_check():
    return {"status": "OK"}


@app.get("/", summary="Root Endpoint", description="Returns a welcome message.")
def read_root():
    return {"message": "Welcome to the Text Generation API"}
