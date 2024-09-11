import re
import base64
import os
import time
import requests as url_requests

from loguru import logger as eval_logger
    

import PIL
import numpy as np

from io import BytesIO
from typing import List, Union
from copy import deepcopy

from decord import VideoReader, cpu
from PIL import Image
try:
    from mistralai import Mistral
except Exception as e:
    eval_logger.debug(f"Please install mistralai packages to use GPT\n{e}")

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model


API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = os.getenv("MISTRAL_API_KEY", None)

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

NUM_SECONDS_TO_SLEEP = 30


@register_model("mistral-api")
class MistralAPI(LMM):
    supported_modalities = ["text-only"]
    def __init__(
        self,
        model_version: str = "mistral-large",
        api_url: str = API_URL,
        timeout: int = 120,
    ) -> None:
    
        self.model_version = model_version
        self.api_url = api_url
        self.client = Mistral(api_key=API_KEY)
        
        self.timeout = timeout
    
    
    def prepare_model(self):            
        self.prepared = True
        eval_logger.info(f"Mistral activated. API_URL: {self.api_url}. MODEL_VERSION: {self.model_version}")
    
    
    def _prepare_generate_kwargs(self, payload, generate_kwargs):
        if "max_new_tokens" in generate_kwargs:
            payload["max_tokens"] = generate_kwargs["max_new_tokens"]
        if "temperature" in generate_kwargs:
            payload["temperature"] = generate_kwargs["temperature"]
        if "top_p" in generate_kwargs:
            payload["top_p"] = generate_kwargs["top_p"]
        if "num_beams" in generate_kwargs:
            payload["num_beams"] = generate_kwargs["num_beams"]
        if "response_format" in generate_kwargs:
            payload["response_format"] = generate_kwargs["response_format"]
        
        return payload

    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]] = None,
        contexts: str = None,
        **generate_kwargs
    ) -> str:
        """
            Call Mistral for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                generate_kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        visuals = None
        messages = []
            
        
        response_json = {"role": "user", "content": []}
        
        messages.append(deepcopy(response_json))
        
        messages[0]["content"].append({"type": "text", "text": contexts})
        
        
        for attempt in range(5):
            try:
                response_data = self.client.chat.complete(
                    model=self.model_version,
                    messages=messages
                )
                response_text = response_data.choices[0].message.content.strip()
                break  # If successful, break out of the loop

            except Exception as e:
                try:
                    error_msg = response_data
                except:
                    error_msg = ""

                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nResponse: {error_msg}.")
                contents = [content for content in messages[0]["content"] if content["type"] == "text"]
                eval_logger.debug(f"Content: {contents}")
                if attempt < 4:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {error_msg}")
                    response_text = ""
        
        return response_text
    
    