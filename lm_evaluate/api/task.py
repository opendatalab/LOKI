import os
import json
import sys
import itertools
import chardet

import lm_evaluate.utils as utils

import datasets

from loguru import logger as eval_logger

from lm_evaluate.api.model import LMM

from abc import ABC, abstractmethod
from tqdm import tqdm


class Task(ABC):
    """
        Abstract implementation of LMM's Task.
    """
    supported_modalities = []
    def __init__(
        self,
        task_name: str,
        dataset_path: str,
        dataset_name: str = None,
        split: str = 'test',
        num_shot: int = 0,
        batch_size: int = 1,
    ):
        """
        Args:
            task_name (str): The name of the defined task, e.g., SynthBench, could be any string.
            dataset_path (str): A path that specifies the path to the dataset data, could be a huggingface path or an absolute path to a json file.
            dataset_name (str, optional): The name of your subtask. Defaults to None.
            split (str, optional):  Split of the dataset, could be 'test' or 'val'. Defaults to 'test'.
            num_shot (int, optional): Few-shot testing. Defaults to 0. Currently only zero-shot is supported.
            batch_size (int, optional): Batch size. Defaults to 1. 
            text_only (bool, optional): Specified whether the data is text_only or multimodal. Defaults to False.
        """
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.num_shot = num_shot
        self.batch_size = batch_size
        
        self.dataset = self.load_dataset()
    
    @abstractmethod
    def doc_to_visual(self, doc):
        """
        A preprocess function for visual data
        """
        pass
    
    @abstractmethod
    def doc_to_text(self, doc):
        """
        A preprocess function for text data
        """
        pass
    
    @abstractmethod
    def doc_to_audio(self, doc):
        """
        A preprocess function for audio data
        """
        
    @abstractmethod
    def doc_to_target(self, doc):
        """
        A preprocess function for answer
        """
        pass
        
    
    def load_dataset(self):
        """
        Currently only supported loading from a json file. # TODO: Add support for huggingface loading.
        """
        
        with open(self.dataset_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)  # 自动检测文件编码
        encoding = result['encoding']  # 检测到的编码
        with open(self.dataset_path, 'r', encoding=encoding) as f:
            dataset = json.load(f)
        
        return dataset
    
    
    @abstractmethod
    def parse_response(self, response):
        """
        Parse response into acceptable predictions for metric
        """
        pass
    
    
    @abstractmethod
    def metric(self, pred, gold):
        """
        Compare predictions with gold answers
        """
        pass
    
    def build_docs(self, limit=None, rank=None, world_size=None) -> None:
        
        eval_logger.info(f"Building docs for task {self.task_name} on rank {rank}")
        
        docs = []
        
        # create a slice of the documents based on rank and world size
        doc_id_iterator = utils.create_iterator([i for i in range(len(self.dataset))], rank, world_size, limit)
        
        doc_id_iterator, doc_id_iterator_counting = itertools.tee(doc_id_iterator)
        
        total_docs = sum(1 for _ in doc_id_iterator_counting)
        
        pbar = tqdm(total=total_docs, desc=f"Building docs on rank {rank}")
        
        for doc_id in doc_id_iterator:
            
            doc = self.dataset[doc_id]
            doc["doc_id"] = doc_id
            
            if not isinstance(doc, list):
                doc = [doc]
                
            docs.extend(doc)
            pbar.update(1)
        
        pbar.close()
        
        return docs
    
    
    @abstractmethod
    def process_responses(self, responses: list):
        """
            Return a result_dict for logging
        """
        pass
    
    @abstractmethod
    def aggregate_results(self, results: list):
        pass
    
    @abstractmethod
    def log_results(self, results, log_file):
        pass
    
    
    @abstractmethod
    def log_accuracies(self, responses, log_file):
        pass
    
    
    @abstractmethod
    def print_pretty_accuracy(self, accuracies):
        pass
    
    
    def evaluate_from_predictions(self, responses):
        results = self.process_responses(self.dataset, responses)
            
        accuracies = self.aggregate_results(results)
        
        self.print_pretty_accuracy(accuracies)
        
        return results, accuracies
    
    
    def evaluate(
        self, 
        model: LMM,
        predict_only: bool = False, 
        batch_size: int = 1,
        **model_kwargs,
    ) -> list:
        """
        Args:
            model (LMM): Models to be evaluated
        
        TODO: Support multi-gpu inference
        """
        if not model.prepared:
            eval_logger.warning(f"Model {model} not prepared. Preparing model here...")
            model.prepare_model()
        
        responses = []
        
        # building docs on GPU:{rank} and {world size}
        docs = self.build_docs(rank=model.rank, world_size=model.world_size)
        # split docs to chunks according to batch size however, there are models that do not support batching, for those models, we reset batch size to 1
        
        if not model.support_batching and batch_size > 1:
            eval_logger.warning(f"Model type: {type(model)} does not support batching. Setting batch_size from {batch_size} to 1...")
            batch_size = 1
            
        chunks = utils.split_chunks(docs, batch_size)
        
        pbar = tqdm(total=len(self.dataset), desc="Model Responding", disable=model.rank != 0)
        
        for doc in self.dataset:
            contexts = self.doc_to_text(doc)
            visuals = self.doc_to_visual(doc)
            if "text-only" in model.supported_modalities and len(model.supported_modalities) == 1:
                visuals = []
            response = model.generate(
                visuals=visuals,
                contexts=contexts,
                **model_kwargs
            )
            
            eval_logger.debug(f"Model response: {response}")

            responses.append(response)
            pbar.update(1)
            
        if model.world_size > 1 and getattr(model, "accelerator", None) is not None:
            model.accelerator.wait_for_everyone()
        
        # Now, we need to reorder the responses to the correct order so that each response is index-aligned with the dataset
        if not predict_only:
            results = self.process_responses(self.dataset, responses)
            
            accuracies = self.aggregate_results(results)
            
            self.print_pretty_accuracy(accuracies)
        
        else:
            results = {
                "model": model.model_version.split("/")[-1],
                "responses": responses
            }
            accuracies = {}
        
        
        return results, accuracies