import re
import base64
import os
import time
import requests as url_requests

from datetime import timedelta
from loguru import logger as eval_logger
    

import PIL
import numpy as np
import torch

from io import BytesIO
from typing import List, Union
from copy import deepcopy

from decord import VideoReader, cpu
from PIL import Image
try:
    from mistralai import Mistral
except Exception as e:
    eval_logger.debug(f"Please install mistralai packages to use Mistral\n{e}")

from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model


API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = os.getenv("MISTRAL_API_KEY", None)

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

NUM_SECONDS_TO_SLEEP = 30


@register_model("mistral-api")
class MistralAPI(LMM):
    supported_modalities = ['text']
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
        self.prepare_model()
    
    
    def prepare_model(self):
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self.accelerator = accelerator        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": 1,
                    "train_batch_size": 1 * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            
            
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1
                
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
    
    