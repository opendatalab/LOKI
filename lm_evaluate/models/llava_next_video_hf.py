import json
import re

import torch

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


try:
    import av
except Exception as e:
    eval_logger.debug("av is required for llava next video")
    
    
try:
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
except ImportError:
    eval_logger.debug("Please upgrade your transformers to support llava-next-video-hf")
from PIL import Image


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


@register_model("llava-next-video-hf")
class LLaVANeXTVideoHF(LMM):
    supported_modalities = ["video", "image", "text"]
    def __init__(
        self,
        model_version: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        max_frame_num: int = 8,
        device: str = "cuda",
        device_map: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        use_flash_attention_2: Optional[bool] = False,
        truncation: bool = True
    ):
        self.model_version = model_version
        self.max_frame_num = max_frame_num
        
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = dtype
        self.truncation = truncation
        
        self.use_flash_attention_2 = use_flash_attention_2
        
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
        
        self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(self.model_version, torch_dtype=self.dtype, use_flash_attention_2=self.use_flash_attention_2, device_map=self.device_map)
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_version)
        self.processor.tokenizer.padding_side = "left"
        
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
    
    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]],
        contexts: str,
        **kwargs
    ) -> str:
        """
            Call llava for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        
        videos = []
        images = []
        
        # four scenarios:
        # visuals is a list of string or PIL image
        # visuals is a string
        # visuals is a PIL Image
        # visuals is None
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    container = av.open(visual)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / self.max_frame_num).astype(int)
                    clip = read_video_pyav(container, indices)
                    eval_logger.debug(clip.shape)
                    videos.append(clip)
                elif isinstance(visual, Image.Image):
                    images.append(visual)
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            container = av.open(visuals)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / self.max_frame_num).astype(int)
            clip = read_video_pyav(container, indices)
            eval_logger.debug(clip.shape)
            videos.append(clip)
        elif isinstance(visuals, Image.Image):
            images.append(visuals)
        
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
        
        for context in contexts:
            if context == "<video>":
                messages[0]["content"].append({"type": "video"})
            elif context == "<image>":
                messages[0]["content"].append({"type": "image"})
            else:
                messages[0]["content"].append({"type": "text", "text": context})
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        eval_logger.debug(f"Prompt: {prompt}")
        
        
        inputs = self.processor(
            text=prompt, 
            images=images if len(images) > 0 else None, 
            videos=videos if len(videos) > 0 else None, 
            return_tensors="pt", 
            truncation=self.truncation
        ).to(self._device, self._model.dtype)
        
        output_ids = self.model.generate(
            **inputs,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=100
            # TODO: add generational kwargs later
        )
        eval_logger.debug(inputs['pixel_values_videos'].size())
        
        for input_id, output_id in zip(inputs['input_ids'], output_ids):
            output = output_id[len(input_id):]
            
        response = self.processor.decode(output, skip_special_tokens=True)
        
        return response
        
        
        
        
        
        