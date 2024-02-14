import os
import numpy as np
import pandas as pd
import sys
import json

import globals as gl

STANDARDIZED_PARAMS_JSON_PATH = gl.STANDARDIZED_PATH + '/standardization_params.json'

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print("CSV file not found.")
        print(e)
        return pd.DataFrame()
    except pd.errors.EmptyDataError as e:
        print("CSV file is empty.")
        print(e)
        return pd.DataFrame()
    except Exception as e:
        print("An error occurred while reading the CSV file.")
        print(e)
        return pd.DataFrame()

def get_column_titles(file_path):
    df = read_csv_file(file_path)
    if df.empty:
        return []
    
    return df.columns.tolist()
    
def strip_csv_column(file_path, column_name):
    df = read_csv_file(file_path)
    if df.empty:
        print(f"strip_csv_column::File {file_path} is empty.")
        return pd.DataFrame()
    
    # Strip the column
    df[column_name] = df[column_name].str.strip()
    return df
    
if __name__ == "__main__":
 
    current_folder_path = gl.STANDARDIZED_PATH
    #for given path get all csv files with PREPROCESSED_POSTFIX
    csv_files = [f for f in os.listdir(current_folder_path) if f.endswith(gl.STANDARDIZED_POSTFIX + '.csv')]
    
    columns_titles = get_column_titles(current_folder_path + '/' + csv_files[0])
    #drop time title
    columns_titles = columns_titles[1:]
    columns_titles = columns_titles[1:]
    
    standardization_params = {}
    total_files = len(csv_files)
    
    print("Starting standardization param calculation...")
    for col in columns_titles:
        standardization_params[col] = {}
        standardization_params[col]['mean'] = 0.00
        standardization_params[col]['std'] = 0.00
        
        for i, file in enumerate(csv_files):
            df = read_csv_file(current_folder_path + '/' + file)
            standardization_params[col]['mean'] += df[col].mean()
            standardization_params[col]['std'] += df[col].std()
        
        standardization_params[col]['mean'] = round(standardization_params[col]['mean'] / total_files, 5)
        standardization_params[col]['std'] = round(standardization_params[col]['std'] / total_files, 5)
        
        print(f"Modality '{col}' - Mean: {standardization_params[col]['mean']:.2f}, Std: {standardization_params[col]['std']:.2f}")
        #write standardization_params to json file
        with open('standardization_stats.json', 'w') as file:
           json.dump(standardization_params, file, indent=4)
    
    print("Standardization finished.")
    
    