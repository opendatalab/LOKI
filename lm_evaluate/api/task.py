import os
import json
import sys
import itertools
import chardet

import lm_evaluate.utils as utils

import datasets
import torch.distributed as dist

from lm_evaluate.api.model import LMM

from loguru import logger as eval_logger

from abc import ABC, abstractmethod
from tqdm import tqdm


class Task(ABC):
    """
        Abstract implementation of LMM's Task.
    """
    def __init__(
        self,
        task_name: str,
        task_modality: str,
        dataset_path: str,
        dataset_name: str = None,
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
        self.task_modality = task_modality
        
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
        result = chardet.detect(raw_data)  
        encoding = result['encoding']
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
    
    
    def process_responses(self, docs: list, responses: list, rank=0):
        """
        Process the LMM's responses.
        1. Parse the response according to the doc
        2. Select metric for the response
        3. Compute accuracy
        4. Return a result_dict, which contains information for logging

        Args:
            docs: the original dataset
            responses: LMM's responses
        """

        results = []
        
        pbar = tqdm(total=len(docs), desc="Processing Results", disable=rank != 0)
        
        for response, doc in zip(responses, docs):
            
            target = self.doc_to_target(doc)
            
            parsed_response = self.parse_response(doc, response)
            accuracy = self.metric(doc, parsed_response, target)
            
            
            result = {
                "parsed_response": parsed_response, 
                "accuracy": accuracy, 
                "contexts": self.doc_to_text(doc),
                "response": response, 
                "target": target, 
                "doc": doc
            }
            
            results.append(result)
            pbar.update(1)
            
        return results
    
    
    def batch_evaluate(
        self, 
        model: LMM,
        predict_only: bool = False, 
        batch_size: int = 1,
        **model_kwargs,
    ):
        """
        Args:
            model (LMM): Models to be evaluated
        
        TODO: Support multi-gpu inference
        """
        
        responses = []
        
        # building docs on GPU:{rank} and {world size}
        docs = self.build_docs(rank=model.rank, world_size=model.world_size)
        # split docs to chunks according to batch size however, there are models that do not support batching, for those models, we reset batch size to 1
        
        if not model.support_batching and batch_size > 1:
            eval_logger.warning(f"Model type: {type(model)} does not support batching. Setting batch_size from {batch_size} to 1...")
            batch_size = 1
            
        chunks = tuple(utils.split_chunks(docs, batch_size))
        
        pbar = tqdm(total=len(chunks), desc="Model Responding", disable=model.rank != 0)
        
        all_response = []
        doc_ids = []
        for chunk in chunks:
            batched_contexts = [self.doc_to_text(doc) for doc in chunk]
            batched_visuals = [self.doc_to_visual(doc) for doc in chunk]
        
            # visuals now is a 2d list, we should leave it to the model to decide whether they want to flatten it
            if "text-only" in model.supported_modalities and len(model.supported_modalities) == 1:
                batched_visuals = []
            
            responses = model.batch_generate(
                batched_visuals=batched_visuals,
                batched_contexts=batched_contexts,
                **model_kwargs
            )
            
            # response is a list
            all_response.extend(responses)
            doc_ids.extend([doc["doc_id"] for doc in chunk])
            
            pbar.update(1)

        responses_with_doc_ids = list(zip(all_response,  doc_ids))
        
        results = self.process_responses(docs, all_response, rank=model.rank)
        
        results_with_doc_ids = list(zip(results, doc_ids))
        
        model.accelerator.wait_for_everyone()
        
        # maybe we should compute results on different gpus, and then aggregate them, this could be faster.
        
        if model.world_size > 1:
            gathered_responses_with_ids = None
            gathered_results_with_ids = None
            
            if model.rank == 0:
                gathered_responses_with_ids = [None] * model.world_size
                gathered_results_with_ids = [None] * model.world_size
            
            dist.gather_object(
                responses_with_doc_ids, 
                object_gather_list=gathered_responses_with_ids if model.rank == 0 else None, 
                dst=0
            )
            
            dist.gather_object(
                results_with_doc_ids,
                object_gather_list=gathered_results_with_ids if model.rank == 0 else None,
                dst=0
            )
            
            if model.rank == 0:
                flattened_responses_with_ids = [item for sublist in gathered_responses_with_ids for item in sublist]
                flattened_responses_with_ids.sort(key=lambda x: x[1])
                flattened_responses = [response for response,  _ in flattened_responses_with_ids]
                
                flattened_results_with_ids = [item for sublist in gathered_results_with_ids for item in sublist]
                flattened_results_with_ids.sort(key=lambda x: x[1])
                flattened_results = [result for result,  _ in flattened_results_with_ids]
                
            else:
                flattened_responses = []
        else:
            flattened_responses = all_response

        if model.rank == 0:
            if not predict_only:
                accuracies = self.aggregate_results(flattened_results)
                self.print_pretty_accuracy(accuracies)
            else:
                results = {
                    "model": model.model_version.split("/")[-1],
                    "responses": flattened_responses
                }
                accuracies = {}
        else:
            results = {}
            accuracies = {}
        
        model.accelerator.wait_for_everyone()

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
        
        responses = []
        
        # building docs on GPU:{rank} and {world size}
        docs = self.build_docs(rank=model.rank, world_size=model.world_size)
        # split docs to chunks according to batch size however, there are models that do not support batching, for those models, we reset batch size to 1
        
        if not model.support_batching and batch_size > 1:
            eval_logger.warning(f"Model type: {type(model)} does not support batching. Setting batch_size from {batch_size} to 1...")
            batch_size = 1
            
        pbar = tqdm(total=len(docs), desc="Model Responding", disable=model.rank != 0)
        
        doc_ids = []
        for doc in docs:
            contexts = self.doc_to_text(doc)
            
            inputs = {"contexts": contexts}
            if self.task_modality == 'image' or self.task_modality == 'video' or self.task_modality == '3D':
                visuals = self.doc_to_visual(doc)
                inputs["visuals"] = visuals
            else:
                inputs["visuals"] = []
            
            inputs.update(model_kwargs)
            response = model.generate(**inputs)
            
            doc_ids.append(doc["doc_id"])
            eval_logger.debug(f"Model response: {response}")

            responses.append(response)
            pbar.update(1)
        
        model.accelerator.wait_for_everyone()
        
        responses_with_doc_ids = list(zip(responses, doc_ids))
        
        results = self.process_responses(docs, responses, rank=model.rank)
        
        results_with_doc_ids = list(zip(results, doc_ids))
        
        if model.world_size > 1:
            model.accelerator.wait_for_everyone()
            gathered_responses_with_ids = None
            if model.rank == 0:
                gathered_responses_with_ids = [None] * model.world_size
                
                gathered_results_with_ids = [None] * model.world_size

            dist.gather_object(
                responses_with_doc_ids, 
                object_gather_list=gathered_responses_with_ids, 
                dst=0
                )
            
            dist.gather_object(
                results_with_doc_ids,
                object_gather_list=gathered_results_with_ids if model.rank == 0 else None,
                dst=0
            )
            
            if model.rank == 0:
                flattened_responses_with_ids = [item for sublist in gathered_responses_with_ids for item in sublist]
                flattened_responses_with_ids.sort(key=lambda x: x[1])
                flattened_responses = [response for response,  _ in flattened_responses_with_ids]
                
                flattened_results_with_ids = [item for sublist in gathered_results_with_ids for item in sublist]
                flattened_results_with_ids.sort(key=lambda x: x[1])
                flattened_results = [result for result,  _ in flattened_results_with_ids]
                
            else:
                flattened_responses = []
                flattened_results = []
        else:
            flattened_responses = responses
            flattened_results = results

        if model.rank == 0:
            if not predict_only:
                accuracies = self.aggregate_results(flattened_results)
                self.print_pretty_accuracy(accuracies)
            else:
                flattened_results = {
                    "model": model.model_version.split("/")[-1],
                    "responses": flattened_responses
                }
                accuracies = {}
        else:
            flattened_results = {}
            accuracies = {}
        
        model.accelerator.wait_for_everyone()
        
        return flattened_results, accuracies