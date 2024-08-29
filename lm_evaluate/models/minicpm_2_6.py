import json
import re

import torch
from decord import VideoReader, cpu
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
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from PIL import Image


@register_model("minicpm-v2.6")
class MiniCPM(LMM):
    supported_modalities = ["video-text", "image-text", "text-only"]
    def __init__(
        self,
        model_version: str = "openbmb/MiniCPM-V-2_6",
        device: str = "cuda",
        device_map: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        attn_implementation: Optional[str] ='sdpa',
        max_num_frames: Optional[int] = 64,
        truncation: bool = True
    ):
        self.model_version = model_version
        
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = dtype
        
        if attn_implementation == "eager":
            eval_logger.warning("Attention implementation: eager is not supported for MiniCPMV2.6")
            if torch.__version__ >= "2.1.2":
                attn_implementation = "sdpa"
            elif is_flash_attn_2_available():
                attn_implementation = "flash_attn_2"
            else:
                attn_implementation = None
            eval_logger.warning(f"Setting attention implementation to {attn_implementation}")
            
        self.attn_implementation = attn_implementation
        self.max_num_frames = max_num_frames
        self.truncation = truncation
        
        
        self.prepare_model()
        
    
    def prepare_model(self):
        
        # accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        # accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # elif accelerator.num_processes == 1 and self.device_map == "auto":
        #     self._device = torch.device(self._device)
        #     self.device_map = self.device_map
        # else:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self._model = AutoModel.from_pretrained(self.model_version, torch_dtype=self.dtype, attn_implementation=self.attn_implementation, trust_remote_code=True).cuda()
        self._model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        
        
        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
        #     # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
        #     # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
        #     if accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         kwargs = {
        #             "train_micro_batch_size_per_gpu": 1,
        #             "train_batch_size": 1 * accelerator.num_processes,
        #         }
        #         AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
        #         eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
        #     if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         self._model = accelerator.prepare(self.model)
        #     else:
        #         self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # elif accelerator.num_processes == 1 and self.device_map == "auto":
        #     eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
        #     self._rank = 0
        #     self._word_size = 1
        # else:
        #     eval_logger.info(f"Using single device: {self._device}")
        #     self._model.to(self._device)
        #     self._rank = 0
        #     self._world_size = 1
        
        self.prepared = True
        
    
    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    
    def encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.max_num_frames:
            frame_idx = uniform_sample(frame_idx, self.max_num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames
    
    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]],
        contexts: str,
        **kwargs
    ) -> str:
        """
            Call mplug-owl for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        
        images = []
        videos = []
        
        # four scenarios:
        # visuals is a list of string or PIL image
        # visuals is a string
        # visuals is a PIL Image
        # visuals is None
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    frames = self.encode_video(visual)
                    videos.append(frames)
                elif isinstance(visual, Image.Image):
                    images.append(visual)
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            videos.append(self.encode_video(visuals))
        elif isinstance(visuals, Image.Image):
            images.append(visuals)
        
        # Segment the text according to video and image token
        
        if len(images) > contexts.count('<image>'):
            eval_logger.warning("<image> tokens num is less than actual number of images. Appending <image> at the front.")
            contexts = "<image> " * (len(images) - contexts.count("<image>")) + contexts

        if len(videos) > contexts.count("<video>"):
            eval_logger.warning("<video> tokens num is less than actual number of images. Appending <video> at the front.")
            contexts = "<video> " * (len(videos) - contexts.count("<video>")) + contexts
        
        # Segment the text according to video and image token
        contexts = re.split(r'(<video>|<image>)', contexts)
        contexts = [context for context in contexts if context]
        
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        video_idx = 0
        image_idx = 0
        for context in contexts:
            if context == "<video>":
                messages[0]["content"] += videos[video_idx]
            elif context == "<image>":
                messages[0]["content"] += images[image_idx]
            else:
                messages[0]["content"] += [context]
        messages.append({"role": "assistant", "content": ""})

        eval_logger.debug(f"Message content: {messages[0]['content']}")
        
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448
        
        params['top_p'] = 1
        params['top_k'] = None
        params['temperature'] = 0
        params['do_sample'] = False
        params['repetition_penalty'] = None
        
        response = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            **params
        )
        
        eval_logger.debug(response)
        
        return response
        
        
        
        
        
        