import os
from datetime import datetime
import json
import yaml

import pandas as pd

from itertools import islice
from collections import defaultdict


def create_iterator(raw_iterator, rank, world_size, limit=None):
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


def make_table(result_dict, all_headers=["Type", "Num", "Accuracy"]):
    """Generate table of results."""
    """
    for subset, subset_result in evaluation_result.items():
        printable_results[subset] = {
            "num": int(subset_result["num_example"]),
            "acc": round(subset_result["acc"], 5),
        }
    """
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    # Set column alignments for LaTeX
    latex_writer.column_alignments = ["center"] * len(all_headers)

    # Set padding for LaTeX columns (this will add space between columns)
    latex_writer.column_format = " ".join(["|c"] * len(all_headers)) + "|"

    values = []

    for subset, subset_result in result_dict.items():
        values.append([subset, subset_result['num'], subset_result['accuracy']])
            
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # Print LaTeX table to see how it looks
    # print(latex_writer.dumps())

    # Return Markdown table (note: column width and text alignment may not be supported)
    return md_writer.dumps()


def get_to_dict_bottom(accuracies, accuracy_for_table, last_key=None):
    """
    Recursively traverse the dictionary to find the bottom dictionaries 
    that contain 'num' and 'accuracy' as keys, and store them in a 
    flat structure for further processing.
    """
    for key in accuracies.keys():
        if isinstance(accuracies[key], dict):
            get_to_dict_bottom(accuracies[key], accuracy_for_table, key)
        else:
            if last_key not in accuracy_for_table.keys():
                accuracy_for_table[last_key] = {key: accuracies[key]}
            else:
                accuracy_for_table[last_key][key] = accuracies[key]
            

def accuracies_to_tables_from_log_dir(log_dir, result_dir):
    """
    Log dir contains many subdirs and files:
        1. Level one: model_type + task_type (separated by "_")
            2. Level two: time of the log
                3. Level three: 
                        result_file: model_name + task_name + "result" (separated by "_")
                        accuracy_file: model_name + task_name + "accuracy" (separated by "_")

    We need to first traverse the log_dir to obtain accuracies of each model and categorize them by the task_type and task_name. If there are two accuracies with different dates, use the latest.
    Then, for each task_type + task_name, we need to make a csv, where the row is the model and the column is the accuracies of that task.
    
    We finally save the csvs to the result_dir
    """
    task_accuracies = defaultdict(lambda: defaultdict(dict))

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith("_accuracy.json"):
                accuracy_file_path = os.path.join(root, file)
                path_parts = os.path.relpath(root, log_dir).split(os.sep)
                if len(path_parts) < 2:
                    continue
                model_task_type, log_time_str = path_parts[0], path_parts[1]
                model_type, task_type = model_task_type.split('_')
                
                log_time = datetime.strptime(log_time_str, "%m%d_%H%M")

                with open(accuracy_file_path, 'r') as f:
                    accuracy_data = json.load(f)
                
                model_name = accuracy_data['model_config']['init_kwargs']['model_version']
                task_name = accuracy_data['task_config']['init_kwargs']['task_name']
                accuracy_for_table = {}
                
                for key in accuracy_data.keys():
                    if 'accuracy' in key:
                        get_to_dict_bottom(accuracy_data[key], accuracy_for_table, key)

                if model_name in task_accuracies[task_type][task_name]:
                    existing_time, _ = task_accuracies[task_type][task_name][model_name]
                    if log_time > existing_time:
                        task_accuracies[task_type][task_name][model_name] = (log_time, accuracy_for_table)
                else:
                    task_accuracies[task_type][task_name][model_name] = (log_time, accuracy_for_table)

    for task_type, task_models in task_accuracies.items():
        for task_name, models in task_models.items():
            df_list = []

            for model_name, (_, accuracies) in models.items():
                accuracies['model'] = model_name  
                df_list.append(pd.DataFrame([accuracies]))

            df = pd.concat(df_list, ignore_index=True)
            df = df[['model'] + [col for col in df.columns if col != 'model']]

            csv_filename = f"{task_type}_{task_name}_accuracies.csv"
            os.makedirs(os.path.join(result_dir, 'csv'), exist_ok=True)
            df.to_csv(os.path.join(result_dir, 'csv', csv_filename), index=False)
            print(df)
            
            markdown_name = f"{task_type}_{task_name}_accuracies.md"
            markdown_table = df.to_markdown(index=False)
            os.makedirs(os.path.join(result_dir, 'markdown'), exist_ok=True)
            with open(os.path.join(result_dir, 'markdown', markdown_name), 'w') as f:
                f.write(markdown_table)


def load_model_from_config(config_path):
    from lm_evaluate.api import MODEL_REGISTRY
    model_config = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    
    
    model_init_kwargs = model_config["init_kwargs"]
    model_type = model_config["model_type"]
    model = MODEL_REGISTRY[model_type](**model_init_kwargs)
    generate_kwargs = model_config["generate_kwargs"]
    
    return model, generate_kwargs, model_config