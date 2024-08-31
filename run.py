import datetime
import os
import json
import yaml
import argparse
import sys

from loguru import logger as eval_logger
from tqdm import tqdm

from lm_evaluate.models import *
from lm_evaluate.tasks import *
from lm_evaluate.api.registry import MODEL_REGISTRY, TASK_REGISTRY


"""
    TODO: Set verbosity -- DONE
    TODO: Allow multi-gpu inference
    TODO: More intricate evaluation results, i.e., we should print a pretty table at the end. -- DONE
    TODO: Add more metrics for synthbench. We need to calculate which generate model the LMM can predict correctly the most. -- DONE
    TODO: Add LLaVA-NeXT-Video -- DONE
    TODO: Add XComposer-2.5 -- DONE
    TODO: Add VILA -- DONE
    TODO: Add internvl2 -- DONE
    TODO: Add MPlug-Owl3 -- DONE
    TODO: Add datetime to log file -- DONE
    TODO: Add generation config for each model
    TODO: Rewrite config logic
"""



def main(args):
    
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if not os.path.exists(args.model_config_path):
        eval_logger.error(f"Model config path: {args.model_config_path} does not exist.")
    
    if not os.path.exists(args.task_config_path):
        eval_logger.error(f"Model config path: {args.task_config_path} does not exist.")
    
    model_config = yaml.load(open(args.model_config_path), Loader=yaml.SafeLoader)
    task_config = yaml.load(open(args.task_config_path), Loader=yaml.SafeLoader)
    
    model_type = model_config["model_type"]
    task_type = task_config["task_type"]
    
    if model_type not in MODEL_REGISTRY:
        eval_logger.error(f"No model named {args.model_name} is found. Supported models: {MODEL_REGISTRY.keys()}")
        sys.exit(-1)
    
    if task_type not in TASK_REGISTRY:
        eval_logger.error(f"No task named {args.task_name} is found. Supported tasks: {TASK_REGISTRY.keys()}")
        sys.exit(-1)
        
    model_init_kwargs = model_config["init_kwargs"]
    print(model_init_kwargs)
    task_init_kwargs = task_config["init_kwargs"]

    model = MODEL_REGISTRY[model_type](**model_init_kwargs)
    task = TASK_REGISTRY[task_type](**task_init_kwargs)
    
    
    model.prepare_model()
    
    model_generate_kwargs = model_config["generate_kwargs"]
    
    results, accuracies = task.evaluate(model, **model_generate_kwargs)
    
    accuracies['model_config'] = model_config
    accuracies['task_config'] = task_config
    
    # get log dir name
    now = datetime.datetime.now()
    datetime_str = now.strftime("%m%d_%H%M")
    
    model_name = model.model_version.split("/")[-1]
    task_name = task.task_name
    file_dir = os.path.join(args.log_dir, f"{args.model_name}_{args.task_name}", datetime_str)
        
    os.makedirs(file_dir, exist_ok=True)
    accuracy_file = os.path.join(file_dir, f"{model_name}_{task_name}_accuracy.json")
    result_file = os.path.join(file_dir, f"{model_name}_{task_name}_result.json")
    
    task.log_accuracies(accuracies, accuracy_file)
    task.log_results(results, result_file)
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--task_config_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--log_dir",
        default="./logs",
        type=str
    )
    parser.add_argument(
        "--logs_prefix",
        default="",
        type=str
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    
    args = parser.parse_args()
    
    main(args)
    
    