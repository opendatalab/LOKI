import re
import base64
import os
import time
import requests as url_requests

import openai
from loguru import logger as eval_logger
import PIL
import numpy as np
import torch

from io import BytesIO
from typing import List, Union
from copy import deepcopy
from datetime import timedelta

from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState
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
    supported_modalities = ["video", "image", "text"]
    def __init__(
        self,
        model_version: str = "gemini-1.5-flash-latest",
        timeout: int = 120
    ) -> None:
    
        self.model_version = model_version
        self.model = genai.GenerativeModel(model_version)
        
        self.timeout = timeout
        self.prepare_model()
    

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
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1:
            self._device = torch.device(self._device)
            self.device_map = self.device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1
        self.prepared = True
        eval_logger.info(f"Gemini activated. MODEL_VERSION: {self.model_version}")

    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]] = None,
        contexts: str = None,
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
    
    