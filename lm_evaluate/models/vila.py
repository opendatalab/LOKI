import json
import re
import copy

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
from PIL import Image

try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.data.dataset import LazySupervisedDataset
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from llava.mm_utils import process_images
except Exception as e:
    eval_logger.debug(f"VILA is not installed. Please install VILA to use this model. Error: {e}")


@register_model("vila")
class VILA(LMM):
    supported_modalities = ["video-text", "image-text", "text-only"]
    def __init__(
        self,
        model_version: str = "Efficient-Large-Model/VILA1.5-13b",
        device: str = "cuda",
        device_map: str = "cuda",
        # dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        attn_implementation: Optional[str] = (
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),
        max_num_frames: Optional[int] = 8,
        truncation: bool = True,
        conv_template: str = "v1",
        use_cache=True,
    ):
        self.model_version = model_version
        
        # if isinstance(dtype, str) and dtype != "auto":
        #     dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = torch.float16
        self.attn_implementation = attn_implementation
        self.max_num_frames = max_num_frames
        self.truncation = truncation
        
        self.max_num_frames = max_num_frames
        
        self.conv_template = conv_template
        self.use_cache = use_cache
        
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
        
        model_name = get_model_name_from_path(self.model_version)
        
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(self.model_version, model_name, device_map=self.device_map, attn_implementation=self.attn_implementation)

        self.model.image_processor = self._image_processor

        self._config = self._model.config

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643
        
        self._tokenizer.padding_side = "left"
        
        
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
        
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    
    @property
    def config(self):
        return self._config
    
    @property
    def device(self):
        return self._device
    
    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = np.linspace(0, total_frame_num - 2, self.max_num_frames, dtype=int)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(img) for img in spare_frames]
        
    
    
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
        
        video_frames = []
        
        processed_image_tensors = []
        
        # four scenarios:
        # visuals is a list of string or PIL image
        # visuals is a string
        # visuals is a PIL Image
        # visuals is None
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    frames = self.encode_video(visual)
                    video_frames.append(len(frames))
                    video_tensors = process_images(frames, self.model.image_processor, self._config).to(dtype=torch.float16, device=self.device)
                    eval_logger.debug(video_tensors.size())
                    processed_image_tensors.append(video_tensors)
                elif isinstance(visual, Image.Image):
                    image_tensor = process_images([visual], self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    processed_image_tensors.append(image_tensor)
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            frames = self.encode_video(visuals)
            video_frames.append(len(frames))
            video_tensors = process_images(frames, self.model.image_processor, self._config).to(dtype=torch.float16, device=self.device)
            eval_logger.debug(video_tensors.size())
            processed_image_tensors.append(video_tensors)
        elif isinstance(visuals, Image.Image):
            image_tensor = process_images([visuals], self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            processed_image_tensors.append(image_tensor)
        
        # Segment the text according to video and image token
        
        if len(images) > contexts.count("<image>"):
            eval_logger.warning("<image> tokens num is less than actual number of images. Appending <image> at the front.")
            contexts = "<image> " * (len(images) - contexts.count("<image>")) + contexts

        if len(videos) > contexts.count("<video>"):
            eval_logger.warning("<video> tokens num is less than actual number of images. Appending <video> at the front.")
            contexts = "<video> " * (len(videos) - contexts.count("<video>")) + contexts
        
        # Segment the text according to video and image token
        contexts = re.split(r'(<video>|<image>)', contexts)
        contexts = [context for context in contexts if context]
        
        prompt = ""
        
        video_idx = 0
        
        for context in contexts:
            if context == "<video>":
                prompt += "<image>\n" * video_frames[video_idx]
                video_idx += 1
            elif context == "<image>":
                prompt += "<image>\n"
            else:
                prompt += context
        
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template].copy())
        else:
            conv = conv_templates[self.conv_template].copy()
        

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        eval_logger.debug(f"Prompt: {prompt}")

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        attention_masks = input_ids.ne(pad_token_ids).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        gen_kwargs = {}
        
        if "max_new_tokens" not in kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in kwargs:
            gen_kwargs["temperature"] = 0.2
        if "top_p" not in kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in kwargs:
            gen_kwargs["num_beams"] = 1

            
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=processed_image_tensors,
                attention_mask=attention_masks,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        eval_logger.debug(response)
        
        return response
        
        