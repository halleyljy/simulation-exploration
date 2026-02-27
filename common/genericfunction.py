"""
Generic functions that can be used across multiple projects
"""

import logging
from typing import Any
from json import load, dump
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

class Constant(Enum):
    """
    Records all constants that are used in most cases
    """
    DIRECTORY = 'output_directory'
    SCENARIO = 'scenario_name'
    SEED = 'seed_value'
    MULTIPROCESSING = 'use_multiprocessing'

def list_of_instance_to_dict_of_lists(instances: list[Any]) -> dict[str, list[Any]]:
    """ Converts a list of class instances to a dictionary of lists
    ie attributes of the class becomes the key of the dictionary and 
    the same attribute in each instance is converted to a list """
    
    if not instances:
        return {}
    
    first_dict: dict = instances[0].__dict__
    key_order: list = list(first_dict.keys())
    return {key: [mydict.__dict__[key] for mydict in instances] for key in key_order}

def aggregate_all_replications(all_runs_results: list[list[dict]]) -> dict[str, list]:
    """
    Aggregates results from multiple simulation replications into a single structure.
    
    Args:
        all_runs_results: A list of the lists returned by run_one_replication.
                         Format: [[{'file1': [...]}, {'file2': [...]}], ... ]
    
    Returns:
        A dictionary where each filename maps to a combined list of data 
        found across all n replications.
    """
    # Use a temporary dictionary to group data by their keys
    master_collection = {}

    for single_run_output in all_runs_results:
        # Each run_result is a list: [file_data1, file_data_2, ...]
        for record_dict in single_run_output:
            for filename, data_list in record_dict.items():
                if filename not in master_collection:
                    master_collection[filename] = []
                # Combine the data list from this run into the master list
                master_collection[filename].extend(data_list)
    
    return master_collection

def get_difference_between_two_lists(main_list: list[int | float], subtract_list: list[int | float]) -> np.array:
    """
    Subtract each items in the main list with items in the subtract list
    Remove values that are np.nan
    ie main list [3, np.nan, 6, 1] and subtract list [1, 5, 3, np.nan] should remove index 1 and 3 from both list
    main list = [3, 6], subtract list = [1, 3]
    [3 - 1, 6 - 3] = [2, 3]
    """
    main_array = np.array(main_list)
    subtracted_array = np.array(subtract_list)
    non_nan_list = ~np.isnan(main_array) & ~np.isnan(subtracted_array)

    return main_array[non_nan_list] - subtracted_array[non_nan_list]

def get_parameter_from_file(filename: Any) -> dict:
    """ Read parameter file from the path given """
    with open(filename, 'r', encoding = 'utf-8') as f:
        return load(f)

def save_parameter_file(params: dict, filename: str) -> None:
    """ save parameter file to the directory given """
    sim_settings = {
        'generated_at': str(datetime.now()),
        'parameters': params
    }

    file_directory: Path = filename
    with open(file_directory, 'w', encoding = 'utf-8') as f:
        dump(sim_settings, f, indent = 4)

def save_to_csv(data: list[dict], filename: str) -> None:
    """ save data given to a csv file with a specific directory and file name """
    df = pd.DataFrame(data)
    df.to_csv(filename, index = False)

def get_logger(name:str, filename: str, debug_level: int) -> Any:
    """ Set logger settings and return logger object """

    # 1. Create logger object
    logger = logging.getLogger(name)
    logger.setLevel(debug_level)

    # 2. Create the FileHandler (corresponds to filename=filename)
    # This handles writing the logs to the file
    handler = logging.FileHandler(filename)
    handler.setLevel(debug_level)

    # 3. Create the Formatter (corresponds to format and datefmt)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )

    # 4. Attach the Formatter to the Handler
    handler.setFormatter(formatter)

    # 5. Attach the Handler to the Logger
    # Check if handlers exist to avoid adding duplicate lines if this function is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
