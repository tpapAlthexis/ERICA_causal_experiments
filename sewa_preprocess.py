import os
import pandas as pd
from scipy.io import arff
import zipfile
import sys
import re
from math import ceil, floor

import globals as gl

ARFF_TO_CSV = False

LLD_frame_n_suffix = "frameIndex"

LLD_COLS_TO_EXCLUDE = ["emotion", "name"]
Aligned_postfix = "_alligned"

LLD_Modality = "ComParE"

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

def get_LLD_df(path):
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
            csv_files = [f for f in os.listdir(lld_path) if f.endswith('.csv') and LLD_Modality in f and Aligned_postfix not in f]
            if len(csv_files) == 0:
                print(f"No csv file with LLD_Modality {LLD_Modality} found in folder {lld_path}. Skipping folder.")
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
            mean_df[LLD_frame_n_suffix] = [i+1 for i in range(len(mean_df))]

            #save as csv using the same name with the postfix '_alligned' to the same folder
            print(f"Saving alligned LLD file to {lld_path}")
            mean_df.to_csv(f"{lld_path}/{LLD_Modality}{Aligned_postfix}.csv", index=False)

        except Exception as e:
            print(f"Error in LLD file: {e}. Skipping file {lld_path}.")


if __name__ == "__main__":
    try:
        arg_path =  gl.SEWA_INPUT_DATA_PATH if len(sys.argv) < 2 else sys.argv[1]
        path = arg_path if os.path.exists(arg_path) else str(gl.CURRENT_DIR_PATH + '/' + arg_path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            sys.exit(1)

        if ARFF_TO_CSV:
            print("----- Start of arff to csv conversion ------")
            convert_arff2csv(arg_path)
            check_arff_csv_pairs(arg_path)
            print("----- End of arff to csv conversion ------")

        get_LLD_df(path)
                
        if not os.path.exists(gl.SEWA_PREPROCESSED_PATH):
            os.makedirs(gl.SEWA_PREPROCESSED_PATH)
    except Exception as e:
        print(f"Error in main: {e}")

    
