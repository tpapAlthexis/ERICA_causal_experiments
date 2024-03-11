import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import globals as gl

import preproc_plotting as pp

def preprocess_data(csv_file):
    print(f"preprocess_data::Initial data size: {len(csv_file)}")

    #check how many missing values are in the dataset
    total_missing = csv_file.isnull().sum().sum()
    csv_file = csv_file.dropna()
    print(f"preprocess_data::Total missing values: {total_missing}. After dropping missing values: {len(csv_file)}")

    ignore_columns = ['frame_path']
    csv_file = csv_file.drop(columns=ignore_columns)

    return csv_file

def time_window_for_step_size(time_window_in_secs, step_size_ms=40):
    if step_size_ms == 0:
        raise ValueError('step_size_ms cannot be 0')
    return int(time_window_in_secs * 1000 / step_size_ms)

def step_size_to_rows(step_size_ms, current_step_size_ms=40):
    if current_step_size_ms == 0:
        raise ValueError('step_size_ms cannot be 0')
    return int(step_size_ms / current_step_size_ms)

def setTimeIndex(df):
    df.rename(columns={'time in seconds': 'time'}, inplace=True)
    df['time'] = pd.to_numeric(df['time'])
    df.set_index('time', inplace=True)
    return df

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        df = setTimeIndex(df)
        
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

def get_samples_csv(path, file_name, save_file=False, gen_histograms=False):
    file_path = path + '/' + file_name
    df = read_csv_file(file_path)
    if df.empty:
        print(f"get_samples_csv::Error reading {file_path}.")
        return False
    
    ignore_columns = (df.filter(regex=gl.ANNOTATIONS_REGEX).columns)
    df = df.drop(columns=ignore_columns)

    try:
        win_sz = time_window_for_step_size(3) #sliding window of 3 seconds
        step_size = step_size_to_rows(400) #step size of 400ms
    except ValueError as e:
        print("get_samples_csv::Error in calculating time window or step size in get_samples_csv.")
        print(e)
        return False

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=win_sz)
    print(f'get_samples_csv::Selected sampling window size: {win_sz} rows and step size: {step_size} rows')
    print(f'get_samples_csv::Current rows: {len(df)}. Expected number of rows: {len(df) // step_size}')

    mean_df = df.rolling(window=indexer, min_periods=1).mean()
    mean_df = mean_df.iloc[::step_size, :]

    if gen_histograms:
        pp.gen_save_histograms(df.drop(columns=['Unnamed: 0']), mean_df.drop(columns=['Unnamed: 0']), 'samples_histograms.pdf')

    if save_file:
        mean_df = mean_df.drop(columns=['Unnamed: 0'])
        savePath = gl.PREPROCESSED_PATH + '/' + file_name.split('.')[0] + gl.PREPROCESSED_POSTFIX + '.csv'
        if not os.path.exists(gl.PREPROCESSED_PATH):
            os.makedirs(gl.PREPROCESSED_PATH)
        mean_df.to_csv(savePath)

    print(f'get_samples_csv::Finished. Total number of rows: {len(mean_df)}')
    return True
    
def get_annotations_csv(path, file_name, save_file=False, gen_histograms=False):
    file_path = path + '/' + file_name
    df = read_csv_file(file_path)
    if df.empty:
        print(f"get_annotations_csv::Error reading {file_path}.")
        return False
    
    #get only annotations
    annotation_columns = (df.filter(regex=gl.ANNOTATIONS_REGEX).columns)
    df = df[annotation_columns]
    
    try:
        win_sz = time_window_for_step_size(3)
        step_size = step_size_to_rows(400)
    except ValueError as e:
        print("get_annotations_csv::Error in calculating time window or step size in get_annotations_csv.")
        print(e)
        return False

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=win_sz)
    print(f'get_annotations_csv::Selected sampling window size: {win_sz} rows and step size: {step_size} rows')
    print(f'get_annotations_csv::Current rows: {len(df)}. Expected number of rows: {len(df) // step_size}')

    mean_df = df.rolling(window=indexer, min_periods=1).mean()
    mean_df = mean_df.iloc[::step_size, :]
    
    # Get the median of each annotation group
    median_x_df = mean_df.filter(regex='_x$').median(axis=1).to_frame('median_arousal')
    median_y_df = mean_df.filter(regex='_y$').median(axis=1).to_frame('median_valence')

    # Combine the median DataFrames
    median_df = pd.concat([median_x_df, median_y_df], axis=1)
    
    if gen_histograms:
        #gen_save_histograms for regex='_x$' columns only
        pp.gen_save_histograms(df.filter(regex='_x$'), mean_df.filter(regex='_x$'), 'annotations_histograms_arousal.pdf', True)
        #gen_save_histograms for regex='_y$' columns only
        pp.gen_save_histograms(df.filter(regex='_y$'), mean_df.filter(regex='_y$'), 'annotations_histograms_valence.pdf', True)
        
        #save median histogram as png
        median_png = pp.create_csv_histograms(median_df)
        for title in median_png[0].keys():
            fig = median_png[0][title]
            fig.savefig(f'{title}_annotation.png')
            plt.close(fig)

    if save_file:
        savePathAnn = gl.PREPROCESSED_PATH + '/' + file_name.split('.')[0] + gl.PREPROCESSED_POSTFIX + '_annotations.csv'
        savePathMedAnn = gl.PREPROCESSED_PATH + '/' + file_name.split('.')[0] + gl.PREPROCESSED_POSTFIX + '_annotations_median.csv'
        if not os.path.exists(gl.PREPROCESSED_PATH):
            os.makedirs(gl.PREPROCESSED_PATH)
        mean_df.to_csv(savePathAnn)
        median_df.to_csv(savePathMedAnn)

    print(f'Finished. Total number of rows: {len(mean_df)}')
    
    return True

if __name__ == "__main__":
    arg_path =  gl.DEFAULT_INPUT_DATA_PATH if len(sys.argv) < 2 else sys.argv[1]
    path = arg_path if os.path.exists(arg_path) else str(gl.CURRENT_DIR_PATH + '/' + arg_path)
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        sys.exit(1)
    
    if not os.path.exists(gl.PREPROCESSED_PATH):
        os.makedirs(gl.PREPROCESSED_PATH)
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    success_count = 0
    failure_count = 0
    failed_files = []

    for file_name in csv_files:
        print(f'-------- Processing {file_name} ----------')
        status1 = get_samples_csv(path, file_name, save_file=True, gen_histograms=False)
        status2 = get_annotations_csv(path, file_name, save_file=True, gen_histograms=False)
        
        if status1 and status2:
            success_count += 1
        else:
            failure_count += 1
            failed_files.append(file_name)

    print('\n---------------------------------')
    print(f'Total successful operations: {success_count}')
    print(f'Total failed operations: {failure_count}')
    print(f'Failed files: {failed_files}')
    print('---------------------------------')
