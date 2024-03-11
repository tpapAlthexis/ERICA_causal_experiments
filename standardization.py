import os
import numpy as np
import pandas as pd
import sys
import json
from sklearn.preprocessing import StandardScaler

import globals as gl

#filter warnings for feature names
import warnings
warnings.filterwarnings('ignore', message="X has feature names, but StandardScaler was fitted without feature names")

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

def standarize_csv(file_path, standardization_params):
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"standarize_csv::File {file_path} is empty.")
        return False
    
    try:
        scaler = StandardScaler()
        
        for col in df.columns:
            if col in standardization_params:
                scaler.mean_ = standardization_params[col]['mean']
                scaler.scale_ = standardization_params[col]['std']
                df[col] = scaler.transform(df[[col]])
        
        savePath = gl.STANDARDIZED_PATH + '/' + os.path.basename(file_path).replace(gl.PREPROCESSED_POSTFIX, gl.STANDARDIZED_POSTFIX)
        if not os.path.exists(gl.STANDARDIZED_PATH):
            os.makedirs(gl.STANDARDIZED_PATH)
        df.to_csv(savePath, index=False)
    except Exception as e:
        print(f"standarize_csv::An error occurred while standardizing the file {file_path}.")
        print(e)
        return False
    
    print(f"standarize_csv::File {file_path} has been standardized and saved to {savePath}.")
    return True
    
if __name__ == "__main__":
    arg_path =  gl.PREPROCESSED_PATH if len(sys.argv) < 2 else sys.argv[1]
    path = arg_path if os.path.exists(arg_path) else str(gl.CURRENT_DIR_PATH + '/' + arg_path)
    print(f"Standardizing files in path: {path}")
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        sys.exit(1)
        
    if not os.path.exists(gl.STANDARDIZED_PATH):
        os.makedirs(gl.STANDARDIZED_PATH)
        
    #for given path get all csv files with PREPROCESSED_POSTFIX
    csv_files = [f for f in os.listdir(path) if f.endswith(gl.PREPROCESSED_POSTFIX + '.csv')]
    
    #filter files which contain NA values
    print(f'Before NA filtering: {len(csv_files)}')
    csv_files = [f for f in csv_files if not read_csv_file(path + '/' + f).empty]
    print(f'After NA filtering: {len(csv_files)}')
    
    if len(csv_files) == 0:
        print("No files to standardize.")
        sys.exit(0)
    
    columns_titles = get_column_titles(path + '/' + csv_files[0])
    #drop time title
    columns_titles = columns_titles[1:]
    
    standardization_params = {}
    total_files = len(csv_files)

    print("Starting standardization param calculation...")
    scaler = StandardScaler()

    for col in columns_titles:
        data = []
        for i, file in enumerate(csv_files):
            df = pd.read_csv(path + '/' + file)
            data.extend(df[col].tolist())
        
        data = pd.DataFrame(data, columns=[col])
        scaler.fit(data)
        
        standardization_params[col] = {}
        standardization_params[col]['mean'] = scaler.mean_[0]
        standardization_params[col]['std'] = np.sqrt(scaler.var_[0])
        
        print(f"Measure '{col}' - Mean: {standardization_params[col]['mean']:.2f}, Std: {standardization_params[col]['std']:.2f}")
        #write standardization_params to json file
        with open(STANDARDIZED_PARAMS_JSON_PATH, 'w') as file:
           json.dump(standardization_params, file, indent=4)
    print("Standardization param calculation finished.")
    
    success_count = 0
    failure_count = 0
    failed_files = []
    
    print("Starting standardization...")
    for file_name in csv_files:
        print(f'-------- Standardizing {file_name} ----------')
        status = standarize_csv(path + '/' + file_name, standardization_params)
        
        if status:
            success_count += 1
        else:
            failure_count += 1
            failed_files.append(file_name)
        print(f'---------------------------------')
        
    print('\n')
    print(f'Total successful operations: {success_count}')
    print(f'Total failed operations: {failure_count}')
    print(f'Failed files: {failed_files}')
    print('---------------------------------')
    print("Standardization finished.")
    
    