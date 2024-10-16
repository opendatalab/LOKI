import json
import re
import os

import torch
import av
import numpy as np

from typing import List, Union, Optional
from copy import deepcopy
from datetime import timedelta

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model


from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


@register_model("xcomposer-2d5")
class XComposer2D5(LMM):
    supported_modalities = ["video", "image", "text"]
    def __init__(
        self,
        model_version: str = "internlm/internlm-xcomposer2d5-7b",
        device: str = "cuda",
        device_map: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        truncation: bool = True,
        tmp_folder: str = './tmp'
    ):
        self.model_version = model_version
        
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = dtype
        self.truncation = truncation
        
        self.tmp_folder = tmp_folder
        
        self.prepare_model()
        
    
    def prepare_model(self):
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and self.device_map == "auto":
            self._device = torch.device(self._device)
            self.device_map = self.device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self._model = AutoModelForCausalLM.from_pretrained(self.model_version, torch_dtype=self.dtype, trust_remote_code=True).cuda()
        self._model.eval()
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.vis_processor = self._model.vis_processor
        self._model.tokenizer = self.tokenizer
        
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
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and self.device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1
        
        self.prepared = True
        
    
    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]],
        contexts: str,
        **kwargs
    ) -> str:
        """
            Call xcomposer for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        
        images = []
        
        # four scenarios:
        # visuals is a list of string or PIL image
        # visuals is a string
        # visuals is a PIL Image
        # visuals is None
        image_idx = 0
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    images.append(visual)
                elif isinstance(visual, Image.Image):
                    visual.save(os.path.join(self.tmp_folder, f"tmp_{image_idx}_{self.rank}_{self.world_size}.jpg"))
                    images.append(os.path.join(self.tmp_folder, f"tmp_{image_idx}_{self.rank}_{self.world_size}.jpg"))
                    image_idx += 1
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            images.append(visuals)
        elif isinstance(visuals, Image.Image):
            visuals.save(os.path.join(self.tmp_folder, f"tmp_{image_idx}_{self.rank}_{self.world_size}.jpg"))
            images.append(os.path.join(self.tmp_folder, f"tmp_{image_idx}_{self.rank}_{self.world_size}.jpg"))
            image_idx += 1
        
        # Segment the text according to video and image token
        prompt = contexts.replace("<video>", "<ImageHere>").replace("<image>", "<ImageHere>")
        
        
        eval_logger.debug(f"Prompt: {prompt}")
        
        try:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    images,
                    
                )
        except Exception as e:
            eval_logger.error(e)
            response = ""
        
        return response
        
        
        
        
        
        