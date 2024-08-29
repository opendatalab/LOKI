import re
import base64
import os
import time
import requests as url_requests

import openai
from loguru import logger as eval_logger
import PIL
import numpy as np

from io import BytesIO
from typing import List, Union
from copy import deepcopy

from decord import VideoReader, cpu
from PIL import Image

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    NUM_SECONDS_TO_SLEEP = 30
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}")
    genai = None


IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

NUM_SECONDS_TO_SLEEP = 30


@register_model("gemini")
class Gemini(LMM):
    supported_modalities = ["video-text", "image-text", "text-only"]
    def __init__(
        self,
        model_version: str = "gemini-1.5-flash-latest",
        timeout: int = 120
    ) -> None:
    
        self.model_version = model_version
        self.model = genai.GenerativeModel(model_version)
        
        self.timeout = timeout
    

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        time.sleep(5)
        return uploaded_obj


    def convert_video(self, images):
        for idx, img in enumerate(images):
            if isinstance(img, str):
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
        return images
    
    def prepare_model(self):
        self.prepared = True
        eval_logger.info(f"Gemini activated. MODEL_VERSION: {self.model_version}")

    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]],
        contexts: str,
        **kwargs
    ) -> str:
        """
            Call gemini for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        if not isinstance(visuals, list) and visuals is not None:
            visuals = [visuals]
        
        if visuals is None:
            visuals = []
        
        message = self.convert_video(visuals) + [contexts]
        
        for attempt in range(5):
            try:
                content = self.model.generate_content(
                    message,
                    # generation_config=config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                response_text = content.text
                break
            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if isinstance(e, ValueError):
                    try:
                        eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                        response_text = ""
                        break
                    except Exception:
                        pass
                if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                    response_text = ""
        
        return response_text
    
    