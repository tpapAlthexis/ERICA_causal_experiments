import os
import pandas as pd
from scipy.io import arff
import zipfile
import sys
import re
from multiprocessing import Pool

import globals as gl

TOTAL_FACIAL_LANDMARKS = 49
TOTAL_EYE_LANDMARKS = 10
NOSE_CENTER_INDEX = 14

ARFF_TO_CSV = False

frame_n_suffix = "frameIndex"
frame_range_suffix = "frame_range"

LLD_COLS_TO_EXCLUDE = ["emotion", "name", "frameTime"]
Aligned_postfix = "_alligned"

Annotations_postfix = "AV_Aligned"

def arff_to_csv(full_path, output_path):
    try:
        content = None
        with open(full_path, "r") as inFile:
            content = inFile.readlines()

        data = False
        header = ""
        new_content = []
        for line in content:
            if not data:
                if "@ATTRIBUTE" in line or "@attribute" in line:
                    attributes = line.split()
                    if("@attribute" in line):
                        attri_case = "@attribute"
                    else:
                        attri_case = "@ATTRIBUTE"
                    column_name = attributes[attributes.index(attri_case) + 1]
                    header = header + column_name + ","
                elif "@DATA" in line or "@data" in line:
                    data = True
                    header = header[:-1]
                    header += '\n'
                    new_content.append(header)
            else:
                new_content.append(line)

        with open(output_path, "w") as outFile:
            outFile.writelines(new_content)
        print(f"Successfully converted arff to csv: {output_path}")
    except Exception as e:
        print(f"Error in arff_to_csv: {e}. Skipping file {full_path}.")

def process_zip(zip_path, extract_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_folder)
        print(f"Successfully extracted zip file: {zip_path}")
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                if file.endswith('.arff'):
                    full_path = os.path.join(root, file)
                    output_path = full_path.replace('.arff', '.csv')
                    arff_to_csv(full_path, output_path)
    except Exception as e:
        print(f"Error in process_zip: {e}")

def convert_arff2csv(directory):
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.arff'):
                    full_arff_path = os.path.join(root, file)
                    full_csv_path = full_arff_path.replace('.arff', '.csv')
                    arff_to_csv(full_arff_path, full_csv_path)
                elif file.endswith('.zip'):
                    zip_path = os.path.join(root, file)
                    extract_folder = os.path.join(root, "extracted_" + os.path.splitext(file)[0])
                    process_zip(zip_path, extract_folder)
    except Exception as e:
        print(f"Error in convert_arff2csv: {e}")

def check_arff_csv_pairs(directory):
    all_converted = True
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.arff'):
                arff_path = os.path.join(root, file)
                csv_path = arff_path.replace('.arff', '.csv')
                if not os.path.exists(csv_path):
                    print(f"Alert: No corresponding .csv file for {arff_path}")
                    all_converted = False
    if all_converted:
        print("All .arff files have been successfully converted to .csv.")

def extract_frame_count(name):
    # This pattern will match two groups of digits at the end of the string, separated by underscores
    match = re.search(r'_(\d+)_(\d+)$', name)
    if match:
        first_number = int(match.group(1))
        second_number = int(match.group(2))
        return second_number - first_number + 1
    else:
        return -1

def align_LLD(path, save=True):
    df = None
    lld_paths = {}
    # for all folders (not the contained ones) in arg_path extract the frame count and collect the LLD paths
    for dir_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir_name)):
            frame_count = extract_frame_count(dir_name)
            if frame_count > 0:
                fpath = os.path.join(path, dir_name)
                dirs_in_fpath = os.listdir(fpath)
                fpath = [os.path.join(fpath, d) for d in dirs_in_fpath if 'LLD' in d and 'extracted' in d]
                if fpath:
                    lld_paths.update({fpath[0]: frame_count})
    if len(lld_paths) == 0:
        print(f"No folders found in path: {path}")
        return None
    
    for lld_path, total_frames in lld_paths.items():
        try:
            #check if a csv file containing LLD_Modality exists otherwise skip this folder and not '_alligned.csv' file
            csv_files = [f for f in os.listdir(lld_path) if f.endswith('.csv') and gl.Measure_Category_Prefixes[gl.AUDIO] in f and Aligned_postfix not in f]
            if len(csv_files) == 0:
                print(f"No csv file with LLD_Modality {gl.Measure_Category_Prefixes[gl.AUDIO]} found in folder {lld_path}. Skipping folder.")
                continue

            lld_data = pd.read_csv(os.path.join(lld_path, csv_files[0]))
            total_LLD = len(lld_data)
            window_size = max(1, round(total_LLD/float(total_frames)))
            remaining_frames = int(total_LLD / window_size - total_frames)
            if remaining_frames > window_size:
                print(f"Error: Remaining frames {remaining_frames} is not in the range [0, window_size]. Skipping folder {lld_path}.")
                continue
           
            #pad or remove the llds with the same value according to remaining_frames
            if remaining_frames < 0:
                padding = pd.DataFrame([lld_data.iloc[-1]] * abs(window_size * remaining_frames), columns=lld_data.columns)
                lld_data = pd.concat([lld_data, padding], ignore_index=True)
            elif remaining_frames > 0:
                lld_data = lld_data[:len(lld_data) - remaining_frames]

            #exclude columns that are not LLDs
            for col in LLD_COLS_TO_EXCLUDE:
                if col in lld_data.columns:
                    lld_data = lld_data.drop(columns=col)
            
            # Create a FixedForwardWindowIndexer with a window size of win_sz
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
          
            # rolling mean with the window size
            mean_df = lld_data.rolling(window=indexer, min_periods=1).mean()

            # get every window_size-th row
            mean_df = mean_df.iloc[::window_size, :]

            #the 'frame_cnt' column will contain increasing numbering from 1 to number of frames
            mean_df[frame_n_suffix] = [i+1 for i in range(len(mean_df))]

            #append to df
            if df is None:
                df = mean_df
            else:
                df = pd.concat([df, mean_df], ignore_index=True)

            #save as csv using the same name with the postfix '_alligned' to the same folder
            if save:
                print(f"Saving alligned LLD file to {lld_path}")
                mean_df.to_csv(f"{lld_path}/{gl.Measure_Category_Prefixes[gl.AUDIO]}{Aligned_postfix}.csv", index=False)

        except Exception as e:
            print(f"Error in LLD file: {e}. Skipping file {lld_path}.")
    
    return df

def getLandmarkPaths(sewa_path):
    try:
        landmark_paths = []
        participant_dirs = os.listdir(sewa_path)
        for dir_name in participant_dirs:
            pat_path = os.path.join(sewa_path, dir_name)
            dirs_in_path = os.listdir(pat_path)
            land_dir_paths = [os.path.join(pat_path, d) for d in dirs_in_path if d.endswith('Landmarks') and d.startswith('extracted')]
            for land_dir_path in land_dir_paths:
                if os.path.isdir(land_dir_path):
                    dirs_in_fpath = os.listdir(land_dir_path)
                    fpath = [os.path.join(land_dir_path, d) for d in dirs_in_fpath if 'Landmarks' in d]
                    if fpath:
                        landmark_paths.append(fpath[0])
        return landmark_paths
    except Exception as e:
        print(f"Error in getLandmarkPaths: {e}")

def process_file(args):
    land_path, txt_file = args
    frameIndex = int(txt_file.split('.')[0])
    with open(os.path.join(land_path, txt_file), 'r') as f:
        lines = [list(map(float, line.strip().split())) for line in f.readlines()[:3]]
    return [frameIndex] + lines[0] + lines[1] + lines[2], len(lines[1]), len(lines[2])

def exportLandmarks(sewa_path):
    landmark_paths = getLandmarkPaths(sewa_path)
    if len(landmark_paths) == 0:
        print(f"exportLandmarks::No Landmark folders found in path: {sewa_path}")
        return None

    for land_path in landmark_paths:
        txt_files = sorted([f for f in os.listdir(land_path) if f.endswith('.txt')])
        with Pool() as p:
            results = p.map(process_file, [(land_path, txt_file) for txt_file in txt_files])
        data = [result[0] for result in results]
        eye_length = results[0][1]
        landmark_length = results[0][2]
        face_cols = ['pitch_deg', 'yaw_deg', 'roll_deg']
        eye_cols = [f'eye_{coord}{i+1}' for i in range(eye_length//2) for coord in ('x', 'y')]
        landmark_cols = [f'landmark_{coord}{i+1}' for i in range(landmark_length//2) for coord in ('x', 'y')]
        df = pd.DataFrame(data, columns=[frame_n_suffix] + face_cols + eye_cols + landmark_cols)

        #save the dataframe as csv to the current path
        df.to_csv(f"{land_path}/Landmarks.csv", index=False)
        print(f"Successfully saved Landmarks.csv to {land_path}")

def normalizeLandmarks(sewa_path, min_max_norm=True):
    try:
        landmark_paths = getLandmarkPaths(sewa_path)
        if len(landmark_paths) == 0:
            print(f"No Landmark folders found in path: {sewa_path}")
            return None

        for land_path in landmark_paths:
            csv_files = [f for f in os.listdir(land_path) if f.endswith('.csv') and 'Landmarks' in f]
            if len(csv_files) == 0:
                print(f"No Landmarks.csv file found in folder {land_path}. Skipping folder.")
                continue
            df = pd.read_csv(os.path.join(land_path, csv_files[0]))

            # Store original nose center coordinates
            nose_center_x = df['landmark_x' + str(NOSE_CENTER_INDEX)].copy()
            nose_center_y = df['landmark_y' + str(NOSE_CENTER_INDEX)].copy()
            
            for i in range(1, TOTAL_FACIAL_LANDMARKS + 1):
                df[f'landmark_x{i}'] -= nose_center_x
                df[f'landmark_y{i}'] -= nose_center_y
            
            for i in range(1, TOTAL_EYE_LANDMARKS + 1):
                df[f'eye_x{i}'] -= nose_center_x
                df[f'eye_y{i}'] -= nose_center_y
            
            if min_max_norm:
                # Calculate min and max values once
                min_x = df.filter(regex='eye_x|landmark_x').min(axis=1)
                max_x = df.filter(regex='eye_x|landmark_x').max(axis=1)
                min_y = df.filter(regex='eye_y|landmark_y').min(axis=1)
                max_y = df.filter(regex='eye_y|landmark_y').max(axis=1)
                
                # Min/Max normalization
                for i in range(1, TOTAL_FACIAL_LANDMARKS + 1):
                    df[f'landmark_x{i}'] = 2 * ((df[f'landmark_x{i}'] - min_x) / (max_x - min_x)) - 1
                    df[f'landmark_y{i}'] = 2 * ((df[f'landmark_y{i}'] - min_y) / (max_y - min_y)) - 1
                for i in range(1, TOTAL_EYE_LANDMARKS + 1):
                    df[f'eye_x{i}'] = 2 * ((df[f'eye_x{i}'] - min_x) / (max_x - min_x)) - 1
                    df[f'eye_y{i}'] = 2 * ((df[f'eye_y{i}'] - min_y) / (max_y - min_y)) - 1
            
            # Save the normalized DataFrame
            df.to_csv(os.path.join(land_path, 'Landmarks_normalized.csv'), index=False)
            print(f"Successfully saved Landmarks_normalized.csv to {land_path}")
    except Exception as e:
        print(f"normalizeLandmarks::An error occurred: {e}")

def get_measure_data(participant_path, data_type, file_condition, folder_name=None):
    data_path = [os.path.join(participant_path, d) for d in os.listdir(participant_path) if data_type in d and 'extracted' in d]
    if not data_path:
        print(f"No {data_type} folder found in path: {participant_path}")
        return None
    # get .csv file
    if folder_name:
        csv_files = [f for f in os.listdir(os.path.join(data_path[0], folder_name + "-" + data_type)) if file_condition(f)]
        data_path[0] = os.path.join(data_path[0], folder_name + "-" + data_type)
    else:
        csv_files = [f for f in os.listdir(data_path[0]) if file_condition(f)]
    if not csv_files:
        print(f"No {data_type} .csv file found in folder {data_path[0]}.")
        return None
    data_df = pd.read_csv(os.path.join(data_path[0], csv_files[0]))
    if data_df.empty:
        print(f"{data_type} File {csv_files[0]} is empty or cannot be read.")
        return None
    return data_df

def getParticipantDF(participant_path):
    try:
        if not os.path.exists(participant_path):
            print(f"Path {participant_path} does not exist.")
            return None

        #get folder name
        folder_name = os.path.basename(participant_path)

        lld_df = get_measure_data(participant_path, 'LLD', lambda f: f.endswith('.csv') and gl.Measure_Category_Prefixes[gl.AUDIO] in f and Aligned_postfix in f)
        landmark_df = get_measure_data(participant_path, 'Landmarks', lambda f: f.endswith('normalized.csv'), folder_name=folder_name)

        #rename landmark columns (except frameIndex) by prepending 'VIDEO_' to the column name
        if landmark_df is not None:
            landmark_df = landmark_df.rename(columns={col: gl.Measure_Category_Prefixes[gl.VIDEO] + '_' + col for col in landmark_df.columns if col != frame_n_suffix})
        #rename audio columns (except frameIndex) by prepending 'ComParE' to the column name
        if lld_df is not None:
            lld_df = lld_df.rename(columns={col: gl.Measure_Category_Prefixes[gl.AUDIO] + '_' + col for col in lld_df.columns if col != frame_n_suffix})

        #check if row count is the same
        if len(lld_df) != len(landmark_df):
            print(f"Row count mismatch between LLD and Landmark dataframes. Skipping participant {folder_name}.")
            return None

        # Merge the two dataframes
        merged_df = pd.merge(lld_df, landmark_df, on=frame_n_suffix)
        return merged_df

    except Exception as e:
        print(f"getParticipantDF::An error occurred: {e}")
        return None

def extract_participant_number(file_path):
    # This regular expression searches for the pattern P followed by digits
    match = re.search(r"P(\d+)", file_path)
    
    if match:
        # Extracts the participant number from the file name
        participant_number = int(match.group(1))
        return participant_number
    else:
        # Returns None if the pattern is not found
        return None

def time_window_for_step_size(time_window_in_secs, step_size_ms=20):
    if step_size_ms == 0:
        raise ValueError('step_size_ms cannot be 0')
    return int(time_window_in_secs * 1000 / step_size_ms)

def step_size_to_rows(step_size_ms, current_step_size_ms=20):
    if current_step_size_ms == 0:
        raise ValueError('step_size_ms cannot be 0')
    return int(step_size_ms / current_step_size_ms)

def subsample_df(df):
    try:
        win_sz = time_window_for_step_size(3) #sliding window of 3 seconds
        step_size = step_size_to_rows(400) #step size of 400ms
    except ValueError as e:
        print("get_samples_csv::Error in calculating time window or step size in get_samples_csv.")
        print(e)
        return False

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=win_sz)
    print(f'subsample_df::Selected sampling window size: {win_sz} rows and step size: {step_size} rows')
    print(f'subsample_df::Current rows: {len(df)}. Expected number of rows: {len(df) // step_size}')

    df_without_suffix = df.drop(columns=[frame_n_suffix])
    mean_df = df_without_suffix.rolling(window=indexer, min_periods=1).mean()
    mean_df = mean_df.iloc[::step_size, :]

    # Add frame_range column
    mean_df[frame_range_suffix] = [f'{i}-{i+win_sz-1}' for i in range(1, len(df)+1, step_size)]
    # Set frame_range col to first
    mean_df = mean_df[[frame_range_suffix] + [col for col in mean_df.columns if col != frame_range_suffix]]

    return mean_df

def export_preprocessed_data(sewa_path):
    try:
        if not os.path.exists(sewa_path):
            print(f"export_preprocessed_data::Path {sewa_path} does not exist.")
            return None
        #get all participant folders
        participant_dirs = [os.path.join(sewa_path, d) for d in os.listdir(sewa_path) if os.path.isdir(os.path.join(sewa_path, d))]
        if len(participant_dirs) == 0:
            print(f"No participant folders found in path: {sewa_path}")
            return None
        
        for p_dir in participant_dirs:
            df = getParticipantDF(p_dir)
            #get dir name
            dir_name = os.path.basename(p_dir)
            participant_number = extract_participant_number(dir_name)
            if participant_number is None:
                print(f"Could not extract participant number from {dir_name}. Skipping participant.")
                continue
            #remove NaN values
            df = df.dropna()
            df = subsample_df(df)

            if df is not None:
                savePath = os.path.join(gl.SEWA_PREPROCESSED_PATH, 'P' + str(participant_number) + gl.PREPROCESSED_POSTFIX + '.csv')
                df.to_csv(savePath, index=False)
                print(f"Successfully saved preprocessed data to {savePath}")
    except Exception as e:
        print(f"Error in export_preprocessed_data: {e}")

def get_annotation_data(p_dir, data_type):
    data_path = [os.path.join(p_dir, d) for d in os.listdir(p_dir) if d.endswith(data_type) and d.startswith('extracted')]
    if not data_path:
        print(f"No {data_type} folder found in path: {p_dir}")
        return None
    # get .csv file
    csv_files = [f for f in os.listdir(data_path[0]) if f.endswith(Annotations_postfix + '.csv')]
    if not csv_files:
        print(f"No {data_type} .csv file found in folder {data_path[0]}.")
        return None
    data_df = pd.read_csv(os.path.join(data_path[0], csv_files[0]))
    if data_df.empty:
        print(f"{data_type} File {csv_files[0]} is empty or cannot be read.")
        return None
    return data_df

def export_annotations(sewa_path):
    if not os.path.exists(sewa_path):
        print(f"export_annotations::Path {sewa_path} does not exist.")
        return None
    #get all participant folders
    participant_dirs = [os.path.join(sewa_path, d) for d in os.listdir(sewa_path) if os.path.isdir(os.path.join(sewa_path, d))]
    if len(participant_dirs) == 0:
        print(f"No participant folders found in path: {sewa_path}")
        return None
    
    for p_dir in participant_dirs:
        arousal_df = get_annotation_data(p_dir, 'Arousal')
        if arousal_df is None:
            continue
        valence_df = get_annotation_data(p_dir, 'Valence')
        if valence_df is None:
            continue
        # Merge the two dataframes
        merged_df = pd.merge(arousal_df, valence_df, on='frame_idx')
        #rename frame_idx to frameIndex
        merged_df = merged_df.rename(columns={'frame_idx': frame_n_suffix})
        #remove NaN values
        merged_df = merged_df.dropna()

        merged_df = subsample_df(merged_df)

        dir_name = os.path.basename(p_dir)
        participant_number = extract_participant_number(dir_name)

        savePath = os.path.join(gl.SEWA_PREPROCESSED_PATH, 'P' + str(participant_number) + '_annotations.csv')
        merged_df.to_csv(savePath, index=False)
        print(f"Successfully saved annotations to {savePath}")

if __name__ == "__main__":
    try:
        arg_path =  gl.SEWA_INPUT_DATA_PATH if len(sys.argv) < 2 else sys.argv[1]
        path = arg_path if os.path.exists(arg_path) else str(gl.CURRENT_DIR_PATH + '/' + arg_path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            sys.exit(1)

        if ARFF_TO_CSV:
            print("----- Start arff to csv conversion ------")
            convert_arff2csv(arg_path)
            check_arff_csv_pairs(arg_path)
            print("----- End arff to csv conversion ------")

        #align_LLD(path)
        exportLandmarks(path)
        normalizeLandmarks(path)
        #export_preprocessed_data(path)
        #export_annotations(path)
                
        if not os.path.exists(gl.SEWA_PREPROCESSED_PATH):
            os.makedirs(gl.SEWA_PREPROCESSED_PATH)
    except Exception as e:
        print(f"Error in main: {e}")
