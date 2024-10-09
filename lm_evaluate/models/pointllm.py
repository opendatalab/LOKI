import json
import re
import sys

import torch
import transformers

import numpy as np

from typing import List, Union, Optional
from copy import deepcopy
from datetime import timedelta
from packaging import version

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model

from decord import VideoReader, cpu
from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from transformers import AutoTokenizer
from PIL import Image

try:
    from pointllm.conversation import conv_templates, SeparatorStyle
    from pointllm.utils import disable_torch_init
    from pointllm.model import *
    from pointllm.model.utils import KeywordsStoppingCriteria
    from pointllm.data import load_objaverse_point_cloud

except Exception as e:
    eval_logger.debug("Please install PointLLM to run pointllm")

    

def init_model(model_path, torch_dtype):
    # Model

    print(f'[INFO] Model name: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch_dtype).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config
    
    if mm_use_point_start_end:
        if "v1" in model_path.lower():
            conv_mode = "vicuna_v1_1"
        else:
            raise NotImplementedError

        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv


@register_model("point-llm")
class PointLLM(LMM):
    supported_modalities = ["point-text", "text-only"]
    support_batching = True
    def __init__(
        self,
        model_version: str = "RunsenXu/PointLLM_13B_v1.2",
        device: str = "cuda",
        device_map: str = "cuda",
        **kwargs
    ):
        self.model_version = model_version
        
        # if isinstance(dtype, str) and dtype != "auto":
        #     dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = torch.float16
        
        self.model_version = model_version
        
        
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
        
        self._model, self._tokenizer, point_backbone_config, self.keywords, self.mm_use_point_start_end, self.conv = init_model(self.model_version, self.dtype)
        
        self.point_token_len = point_backbone_config['point_token_len']
        self.default_point_patch_token = point_backbone_config['default_point_patch_token']
        self.default_point_start_token = point_backbone_config['default_point_start_token']
        self.default_point_end_token = point_backbone_config['default_point_end_token']
        
        eval_logger.debug(point_backbone_config)
        
        self.tokenizer.padding_side = "left"
        
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
        
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    
    @property
    def device(self):
        return self._device
    
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
    
    def generate(
        self, 
        visuals: Union[Image.Image, str, List[Union[Image.Image, str]]],
        contexts: str,
        **kwargs
    ) -> str:
        """
            Call pointllm for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.
            
            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them. 
                contexts: Prompt text.
                kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        
        # two scenarios:
        # visuals is None
        # if visuals is not None, it must be a numpy array of shape (N, 6)
        if visuals is not None:
            assert len(visuals) == 1
            point_clouds = torch.from_numpy(visuals[0]).unsqueeze_(0).to(self.dtype).cuda()
            eval_logger.debug(point_clouds.size())
        
        if self.mm_use_point_start_end:
            qs = self.default_point_start_token + self.default_point_patch_token * self.point_token_len + self.default_point_end_token + '\n' + contexts
        else:
            qs = self.default_point_patch_token * self.point_token_len + '\n' + contexts
        
        conv = self.conv.copy()

        conv.append_message(self.conv.roles[0], qs)
        conv.append_message(self.conv.roles[1], None)
        prompt = conv.get_prompt()
        
        eval_logger.debug(f"\nPrompt: {prompt}")

        inputs = self.tokenizer([prompt])
        
        input_ids = torch.as_tensor(inputs.input_ids).to(torch.long).cuda()
        
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        stop_str = self.keywords[0]
        
        try:
            with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        point_clouds=point_clouds,
                        do_sample=False,
                        temperature=1.0,
                        top_k=None,
                        max_length=2048,
                        top_p=1,
                        stopping_criteria=[stopping_criteria]
            )
        
                
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        
        except Exception as e:
            outputs = ""
        
        
        return outputs
