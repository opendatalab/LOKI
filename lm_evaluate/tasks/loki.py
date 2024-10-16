import ast
import json
import random

import trimesh
import numpy as np

from collections import defaultdict
from typing import Union
from PIL import Image

from tqdm import tqdm

from lm_evaluate.api.registry import register_task
from lm_evaluate.api.task import Task
from lm_evaluate.utils import make_table
from .loki_utils.gpt_as_judge_utils import *

from loguru import logger as eval_logger




__all__ = ["Loki"]


##########################################
#### Helper function for building metrics
##########################################


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def parse_options(options):
    if isinstance(options, str):
        options = ast.literal_eval(options)
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option.replace(option_letter + '.', '').strip()}" for option_letter, option in zip(option_letters, options)])
    return choices_str


TRUE_OR_FALSE_POST_PROMPT = "Answer with strictly Yes or No."
MULTI_CHOICE_POST_PROMPT = "Answer with the option letter."


@register_task("loki")
class Loki(Task):
    def doc_to_visual(self, doc):
        """
        Example:
        """
        if doc['modality'] == 'video-text':
            return doc['video_path']

        elif doc['modality'] == 'image-text':
            images = doc['image_path']
            pil_images = []
            if not isinstance(images, list):
                images = [images]
            for image in images:
                pil_images.append(Image.open(image).convert("RGB"))
            
            return pil_images

        elif doc['modality'] == 'point-text':
            points = doc['point_path']
            
            np_points = []
            if not isinstance(points, list):
                points = [points]
            
            for point in points:
                xyz_data = np.asarray(trimesh.load(point).vertices)
                eval_logger.debug(xyz_data.shape)
                rgb_data = np.zeros((4096, 3))
                
                mesh = np.hstack((xyz_data, rgb_data))
                np_points.append(mesh)
                
            return np_points
        
        else:
            return []

    def doc_to_text(self, doc):
        """
        Example:
        """
        if "cot" in self.task_name:
            return doc["question"]
        
        if doc['metric'] == 'open-ended':
            return f"{doc['question']}\n{TRUE_OR_FALSE_POST_PROMPT}"
        elif doc['metric'] == 'multi-choice':
            return f"{doc['question']}\n{parse_options(doc['choices'])}\n{MULTI_CHOICE_POST_PROMPT}"
        else:
            return doc["question"]
    
    def doc_to_audio(self, doc):
        pass

    def doc_to_target(self, doc):
        """
        Example:
        """
        return doc['answer']
    
    
    def parse_response(self, doc, response):
        if doc['metric'] == 'open-ended':
            return self.parse_true_or_false(response)
        elif doc['metric'] == 'multi-choice':
            index2ans, all_choices = self.parse_multi_choice_info(doc['choices'])
            return self.parse_multi_choice_response(response, all_choices, index2ans, doc)
        
        return response
    
    
    @classmethod
    def parse_multi_choice_response(self, response, all_choices, index2ans, doc):
        """
        Parse the prediction from the generated response.
        Return the predicted index e.g., A, B, C, D.
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
        """
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match

        index_ans = True
        ans_with_brack = False
        ans_with_star = False
        candidates = []
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
        
        for choice in all_choices: # e.g., **A**, **B**, **C**, **D**
            if f"**{choice}**" in response:
                candidates.append(choice)
                ans_with_star = True

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A B C D
                if f"{choice} " in response:
                    candidates.append(choice)

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A. B. C. D.
                if f"{choice}." in response:
                    candidates.append(choice)
        
        if len(candidates) == 0:
            for ans in index2ans.values():
                if f"{ans}" in response:
                    candidates.append(ans)

        # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
        if len(candidates) == 0 and len(response.split()) > 5:
            for index, ans in index2ans.items():
                if ans.lower() in response.lower():
                    candidates.append(index)
                    index_ans = False  # it's content ans.

        if len(candidates) == 0:  # still not get answer, randomly choose one.
            print(all_choices, response, doc)
            pred_index = random.choice(all_choices)
        elif len(candidates) > 1:
            start_indexes = []
            if index_ans:
                if ans_with_brack:
                    for can in candidates:
                        index = response.rfind(f"({can})")
                        start_indexes.append(index)  # -1 will be ignored anyway
                    # start_indexes = [generated_response.index(f'({can})') for can in candidates]
                elif ans_with_star:
                    for can in candidates:
                        index = response.rfind(f"**{can}**")
                        start_indexes.append(index)
                else:
                    for can in candidates:
                        index = response.rfind(f" {can} ")
                        start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.lower().rfind(index2ans[can].lower())
                    start_indexes.append(index)
            # get the last one
            pred_index = candidates[np.argmax(start_indexes)]
        else:  # if only one candidate, use it.
            pred_index = candidates[0]

        return pred_index
    
    
    @classmethod
    def parse_multi_choice_info(self, options):
        """
        Given the list of options for multiple choice question
        Return the index2ans and all_choices
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
        """
        if isinstance(options, str):
            options = ast.literal_eval(options)
        start_chr = "A"
        all_choices = []
        index2ans = {}
        for i, option in enumerate(options):
            index2ans[chr(ord(start_chr) + i)] = option
            all_choices.append(chr(ord(start_chr) + i))
        
        
        return index2ans, all_choices
    
    @classmethod
    def parse_true_or_false(self, pred_ans):
        """Brought from Otter Eval"""
        pred_ans = pred_ans.lower().strip().replace(".", "")
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        elif len(pred_ans) == 1:
            if pred_ans == "y":
                pred_label = "yes"
            elif pred_ans == "n":
                pred_label = "no"
            else:
                pred_label = "other"
        else:
            prefix_pred_ans = pred_ans
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label
    
    
    @classmethod
    def eval_multi_choice(self, gold_i, pred_i):
        """
        Evaluate a multiple choice instance.
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
        """
        correct = False
        # only they are exactly the same, we consider it as correct
        if isinstance(gold_i, list):
            for answer in gold_i:
                if answer == pred_i:
                    correct = True
                    break
        else:  # gold_i is a string
            if gold_i == pred_i:
                correct = True
        return correct
    
    
    @classmethod
    def eval_open(self, gold_i, pred_i):
        """
        Evaluate an open question instance
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
        """
        correct = False
        if isinstance(gold_i, list):
            # use float to avoid trivial matches
            norm_answers = []
            for answer in gold_i:
                norm_answers.extend(normalize_str(answer))
        else:
            norm_answers = normalize_str(gold_i)
        for pred in pred_i:  # pred is already normalized in parse response phase
            if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
                for norm_ans in norm_answers:
                    # only see if the string answer in the string pred
                    if isinstance(norm_ans, str) and norm_ans in pred:
                        if not correct:
                            correct = True
                        break
            else:  # it's a float number
                if pred in norm_answers:
                    if not correct:
                        correct = True
                    break
        return correct
    
    
    def metric(self, doc: dict, pred: Union[str, list[str]], gold: str):
        """
        See if pred matches gold

        Args:
            doc (dict): original doc so that we know which metric to use
            pred (Union[str, list[str]]): Model prediction candidates
            gold (str): Gold answer
        """
        if doc['metric'] == 'open-ended':
            correct = self.eval_open(gold, [pred])
        
        elif doc['metric'] == 'multi-choice':
            correct = self.eval_multi_choice(gold, pred)
        
        elif doc['metric'] == 'image-model-as-judge':
            problems = doc["problems"]
            global_desc = problems["global"][0]["desc"]
            regional_problems = problems["regional"]
            
            image = Image.open(doc["image_path"]).convert("RGB")
            images = []
            regions = [
                (
                    regional_problem["region"][0], 
                    regional_problem["region"][1], 
                    regional_problem["region"][0] + regional_problem["region"][2], 
                    regional_problem["region"][1] + regional_problem["region"][3]
                )  
                for regional_problem in regional_problems
            ]
            region_descs = [regional_problem["desc"] for regional_problem in regional_problems]
            images = [image] + [image.crop(region) for region in regions]
            
            compositional_regional_desc = "\n".join([f"Regionally cropped image: <image>.\nRegional description: {region_desc}" for region_desc in region_descs])
            
            eval_prompt = GPT_EVAL_PROMPT_IMAGE.format(global_desc=global_desc, compositional_regional_desc=compositional_regional_desc, assistant_response=pred)
            
            correct = gpt_eval([("user", eval_prompt)], pred, images)
        
        elif doc["metric"] == '3d-model-as-judge':
            problems = doc["problems"]
            rgb_texture_problems = problems["rgb_texture"]
            normal_lighting_problems = problems["normal_lighting"]
            
            image = Image.open(doc["image_path"]).convert("RGB")
            
            eval_prompt = GPT_EVAL_PROMPT_3D.format(
                rgb_texture_problems=rgb_texture_problems, 
                normal_map_problems=normal_lighting_problems,
                assistant_response=pred
            )
            
            correct = gpt_eval([("user", eval_prompt)], pred, [image], evaluator=ThreeDEvaluator)
        
        elif doc['metric'] == 'video-model-as-judge':
            problems = doc['problems']
            
            # if ""
            segment_annotations = ""
            keyframe_annotations = ""
            global_annotations = ""
            
            key_frames = []
            
            if 'segment_anno' in problems:
                segment_annotations = "\n".join(problems['segment_anno'])
            
            if 'key_frame_anno' in problems:
                keyframe_annotations = "\n".join([f"Key frame: <image>\n Annotations: {key_frame_annotation}" for key_frame_annotation in problems['key_frame_anno']])
                key_frames = [Image.open(key_frame_path) for key_frame_path in problems['key_frame_path']]
            
            if "global_anno" in problems:
                global_annotations = problems["global_anno"]
                
            eval_prompt = GPT_EVAL_PROMPT_VIDEO.format(
                segment_annotations=segment_annotations,
                keyframe_annotations=keyframe_annotations,
                global_annotations=global_annotations,
                assistant_response=pred
            )
            
            video_frames = video_uniform_sampling(doc['video_path'], 16)
            # video_frames = []
            eval_prompt = eval_prompt.replace("<video>", "\n".join([f"video frame {i + 1}: <image>" for i in range(len(video_frames))]))
            
            images = video_frames + key_frames
            
            correct = gpt_eval([("user", eval_prompt)], pred, images, evaluator=VideoEvaluator)
            
            
        else:
            eval_logger.error(f"Metric {doc['metric']} not supported")
            correct = False
        
        return correct
    
    
    def evaluate_random(self):
        responses = []
        
        options = ["A", "B", "C", "D", "E"]
        
        pbar = tqdm(total=len(self.dataset), desc="Model Responding")
        
        for doc in self.dataset:
            if doc['metric'] == 'open-ended':
                responses.append(random.choice(['yes', 'no']))
            elif doc['metric'] == 'multi-choice':
                responses.append(random.choice(options[:len(doc['choices'])]))

            pbar.update(1)
        
        results = self.process_responses(self.dataset, responses)
        accuracies = self.aggregate_results(results)
        self.print_pretty_accuracy(accuracies)
        
        return results, accuracies
    
    
    def log_results(self, results, log_file):
        json.dump(results, open(log_file, 'w'), indent=4, ensure_ascii=False)
    
    
    def log_accuracies(self, results, log_file):
        json.dump(results, open(log_file, 'w'), indent=4, ensure_ascii=False)
        
        
    def aggregate_results(self, results: list):
        """
        We calculate accuracy at two levels: Per metric accuracy and Per question type accuracy
        """
        per_metric_accuracy = defaultdict(list)
        per_question_type_accuracy = defaultdict(list)
        
        accuracy = []
        
        pbar = tqdm(total=len(self.dataset), desc="Aggregating Results")
        
        circular_results = []
        for result in results:
            question_type = result['doc']['question_type']
            metric = result['doc']['metric']
            
            if "model-as-judge" in metric:
                result_accuracy = result['accuracy']
                for key in result_accuracy:
                    per_metric_accuracy[key].append(result_accuracy[key]['score'])
                complete_score = sum([result_accuracy[key]['score'] for key in result_accuracy.keys()])
                accuracy.append(complete_score)
            else:            
                per_metric_accuracy[metric].append(result['accuracy'])
                per_question_type_accuracy[question_type].append(result['accuracy'])
                accuracy.append(result['accuracy'])
            
            if "circular_marker" in result["doc"].keys():
                circular_results.append(result)
                
            pbar.update(1)
        
        if len(circular_results) > 0:
            eval_logger.info(f"Performing circular eval: {self.task_name}")
            
            circular_table = defaultdict(list)
            
            # log circular results
            for result in circular_results:
                result_accuracy = result["accuracy"]
                marker = result["doc"]["circular_marker"]
                
                circular_table[marker].append(result_accuracy)
            
            # compute circular accuracy
            for key, value in circular_table.items():
                question_type = "_".join(key.split("_")[:-1]) + "_circular"
                per_question_type_accuracy[question_type].append(sum(value) == len(value))
        
        for metric in per_metric_accuracy.keys():
            per_metric_accuracy[metric] = {'accuracy': sum(per_metric_accuracy[metric]) / len(per_metric_accuracy[metric]), 'num': len(per_metric_accuracy[metric])}
        
        for question_type in per_question_type_accuracy.keys():
            per_question_type_accuracy[question_type] = {'accuracy': sum(per_question_type_accuracy[question_type]) / len(per_question_type_accuracy[question_type]), 'num': len(per_question_type_accuracy[question_type])}
            
        return {
            "total_accuracy": {"accuracy": sum(accuracy) / len(accuracy), "num": len(accuracy)}, 
            "per_metric_accuracy": per_metric_accuracy, 
            "per_question_type_accuracy": per_question_type_accuracy
        }
    
    def print_pretty_accuracy(self, accuracies):
        accuracy_for_table = {}
        def get_to_dict_bottom(accuracies, last_key=None):
            """
            Assume the final dict has num and accuracy as keys
            """
            for key in accuracies.keys():
                if isinstance(accuracies[key], dict):
                    get_to_dict_bottom(accuracies[key], key)
                else:
                    if last_key not in accuracy_for_table.keys():
                        accuracy_for_table[last_key] = {key: accuracies[key]}
                    else:
                        accuracy_for_table[last_key][key] = accuracies[key]
        get_to_dict_bottom(accuracies)
        table_string = make_table(accuracy_for_table)
        eval_logger.info(f"\n{table_string}")
    
    