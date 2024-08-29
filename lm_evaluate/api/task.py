import os
import json
import sys
import itertools

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
        
        with open(self.dataset_path, 'r') as f:
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
        
        # eval_logger.info(f"Building docs for task {self.task_name} on rank {rank}")
        
        # docs = []
        
        # doc_id_iterator = utils.create_iterator([i for i in range(len(self.datasets)), rank, world_size, limit])
        # doc_id_iterator, doc_id_iterator_counting = itertools.tee(doc_id_iterator)
        
        # total_docs = sum(1 for _ in doc_id_iterator_counting)
        
        # pbar = tqdm(total=total_docs, desc=f"Building docs", disable=(rank != 0))
        
        # for doc_id in doc_id_iterator:
            
        #     doc = self.datasets[doc_id]
            
        #     if not isinstance(doc, list):
        #         doc = [doc]
                
        #     docs.extend(doc)
        #     pbar.update(1)
        
        # pbar.close()
        
        # self._docs = docs
        pass
    
    
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
    
    def evaluate(
        self, 
        model: LMM, 
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
        
        # self.build_docs(rank=model.rank, world_size=model.world_size)
        
        pbar = tqdm(total=len(self.dataset), desc="Model Responding")
        
        for doc in self.dataset:
            contexts = self.doc_to_text(doc)
            visuals = self.doc_to_visual(doc)
            # target = self.doc_to_target(doc)
            
            # res_doc = {}
            # try:
            response = model.generate(
                visuals,
                contexts,
                **model_kwargs
            )
            # except Exception as e:
            #     eval_logger.error(f"Model tried to respond, but ran into: {e}")
            #     response = ""
            
            eval_logger.debug(f"Model response: {response}")
            

            responses.append(response)
            pbar.update(1)

        results = self.process_responses(self.dataset, responses)
        
        accuracies = self.aggregate_results(results)
        
        self.print_pretty_accuracy(accuracies)
        
        # eval_logger.info()
        
        return results, accuracies
            