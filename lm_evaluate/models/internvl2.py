import json
import math
import re

import torch
import av
import numpy as np
import torchvision.transforms as T

from typing import List, Union, Optional
from copy import deepcopy
from datetime import timedelta

from lm_evaluate.api.model import LMM
from lm_evaluate.api.registry import register_model

from decord import VideoReader, cpu
from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from PIL import Image
from torchvision.transforms.functional import InterpolationMode



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    frames = vr.get_batch(frame_indices).asnumpy()
    for frame in frames:
        img = Image.fromarray(frame).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return pixel_values_list, num_patches_list


def load_image(image_file, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    image = image_file.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


@register_model("internvl2")
class InternVL2(LMM):
    supported_modalities = ["video-text", "image-text", "text-only"]
    def __init__(
        self,
        model_version: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda",
        device_map: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        max_num_frames: Optional[int] = 8,
        truncation: bool = True
    ):
        self.model_version = model_version
        
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        self._device = device
        self.device_map = device_map
        self.dtype = dtype
        self.truncation = truncation
        
        self.max_num_frames = max_num_frames
        
        
        self.prepare_model()
        
    
    def prepare_model(self):
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            self._model = AutoModel.from_pretrained(self.model_version, torch_dtype=self.dtype, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).cuda()
            self._model.eval()
        elif accelerator.num_processes == 1 and self.device_map == "auto":
            self._device = torch.device(self._device)
            device_map = split_model(self.model_version.split("/")[1])
            self._model = AutoModel.from_pretrained(self.model_version, torch_dtype=self.dtype, use_flash_attn=True, device_map=device_map, low_cpu_mem_usage=True, trust_remote_code=True)
            self._model.eval()

        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            self._model = AutoModel.from_pretrained(self.model_version, torch_dtype=self.dtype, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).cuda()
            self._model.eval()
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self._model.tokenizer = self.tokenizer
        
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
        self.accelerator = accelerator
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
        
        pixel_values_list = []
        num_patches_lists = []
        
        video_frames = []
        num_images = 0
        # four scenarios:
        # visuals is a list of string or PIL image
        # visuals is a string
        # visuals is a PIL Image
        # visuals is None
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    pixel_values, num_patches_list = load_video(visual, num_segments=self.max_num_frames, max_num=1)
                    pixel_values_list.extend(pixel_values)
                    num_patches_lists.extend(num_patches_list)
                    video_frames.append(len(num_patches_list))
                elif isinstance(visual, Image.Image):
                    pixel_values = load_image(visual, max_num=12)
                    pixel_values_list.append(pixel_values)
                    num_patches_lists.append(pixel_values.size(0))
                    num_images += 1
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            pixel_values, num_patches_lists = load_video(visuals, num_segments=self.max_num_frames, max_num=1)
            pixel_values_list.extend(pixel_values)
            video_frames.append(len(num_patches_lists))
        elif isinstance(visuals, Image.Image):
            pixel_values = load_image(visual, max_num=12)
            num_patches_lists = [pixel_values.size(0)]
            pixel_values_list.append(pixel_values)
            num_images += 1
            
        
        if len(pixel_values_list) > 0:
            pixel_values_list = [pixel_values.to(self.dtype).cuda() for pixel_values in pixel_values_list]
            
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None
    
        if num_images > contexts.count('<image>'):
            eval_logger.warning("<image> tokens num is less than actual number of images. Appending <image> at the front.")
            contexts = "<image> " * (num_images - contexts.count("<image>")) + contexts

        if len(video_frames) > contexts.count("<video>"):
            eval_logger.warning("<video> tokens num is less than actual number of images. Appending <video> at the front.")
            contexts = "<video> " * (len(video_frames) - contexts.count("<video>")) + contexts
            
        contexts = re.split(r'(<video>)', contexts)
        contexts = [context for context in contexts if context]
        
        prompt = ""
        
        video_idx = 0
        for context in contexts:
            if context == "<video>":
                num_frames = video_frames[video_idx]
                prompt += ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
                video_idx += 1
            else:
                prompt += context
        
        # eval_logger.debug(f"Prompt: {prompt}")
        
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_lists,
                history=None, 
                return_history=True
            )
        except Exception as e:
            eval_logger.error(e)
            response = ""
        
        return response
        
        
        
        
        
        