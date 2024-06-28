import pandas as pd
import globals as gl
import numpy as np

def clear_data(data_df):
    drop_col = [col for col in data_df.columns if 'time' in col or 'frame_range' in col or 'Unnamed: 0' in col]
    return data_df.drop(columns=drop_col, errors='ignore')

def readData(participant, dataset):
    data_df = clear_data(pd.read_csv(gl.getParticipantStandardizedPath(participant, dataset)))
    annotations_df = clear_data(pd.read_csv(gl.getAnnotationsPath(participant, dataset)))

    return data_df, annotations_df

def readDataAll_p(measures_list, dataset, exclude_participants=[], data_percentage=1.0, participant_random_indices={}):
    participants = gl.getParticipants(dataset)
    participants = [p for p in participants if p not in exclude_participants]
    participants.sort()

    data = {}
    annotations = []

    #read data and annotations for each measure and for all participants
    for measure in measures_list:
        data_measure = []
        for participant in participants:
            data_measure_df = clear_data(pd.read_csv(gl.getParticipantStandardizedPath(participant, dataset)))
            prefixes = gl.Measure_Category_Prefixes.get(measure, [])
            cols = [col for col in data_measure_df.columns if any(col.startswith(prefix) for prefix in prefixes)]
            data_measure_df = data_measure_df[cols]
            #if current participant random indices are not available, then generate random indices
            if participant not in participant_random_indices:
                #create participant entry in participant_random_indices
                participant_random_indices[participant] = np.random.choice(data_measure_df.shape[0], int(data_percentage * data_measure_df.shape[0]), replace=False)
            
            data_measure_df = data_measure_df.iloc[participant_random_indices[participant]]
            data_measure.append(data_measure_df)
        
        data[measure] = pd.concat(data_measure)

    for participant in participants:
        annotations_df = clear_data(pd.read_csv(gl.getAnnotationsPath(participant, dataset)))
        #use participant_random_indices to get the annotations for the same indices as the data
        annotations_df = annotations_df.iloc[participant_random_indices[participant]]
        annotations.append(annotations_df)

    data_df = pd.concat(data.values(), axis=1)
    annotations_df = pd.concat(annotations)

    #if data rows are not equal to annotations rows, then trigger alert and return null
    if data_df.shape[0] != annotations_df.shape[0]:
        print("Data and annotations rows are not equal")
        return None, None
           
    return data_df, annotations_df

def validate_categorized_data(categorized_data, min_samples=10, min_features=2):
    valid_data = {}
    invalid_categories = []
    
    for category, data in categorized_data.items():
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        elif isinstance(data, np.ndarray):
            data_values = data
        else:
            print(f"validate_categorized_data: Invalid data type for category '{category}'. Expecting pandas DataFrame or numpy array.")
            invalid_categories.append(category)
            continue
        
        if np.any(np.isnan(data_values)):
            nan_indices = np.argwhere(np.isnan(data_values))
            print(f"validate_categorized_data: Data for category '{category}' contains NaN values at indices: {nan_indices}.")
            invalid_categories.append(category)
            continue
        
        if data_values.shape[0] < min_samples:
            print(f"validate_categorized_data: Category '{category}' does not have enough samples ({data_values.shape[0]}). Minimum required is {min_samples}.")
            invalid_categories.append(category)
            continue
        
        if data_values.shape[1] < min_features:
            print(f"validate_categorized_data: Category '{category}' does not have enough features ({data_values.shape[1]}). Minimum required is {min_features}.")
            invalid_categories.append(category)
            continue

        valid_data[category] = data
    
    return valid_data, invalid_categories

# Function to categorize columns
def categorize_columns(df):
    # Categorizes columns based on predefined categories and returns separate dataframes for each.
    categorized = {}
    other_features = set(df.columns)
    for category, prefixes in gl.Measure_Category_Prefixes.items():
        if isinstance(prefixes, list):
            prefixes = tuple(prefixes)  # Convert list to tuple for startswith
        elif isinstance(prefixes, str):
            prefixes = (prefixes,)  # Convert single string to tuple

        category_features = [col for col in df.columns if col.startswith(prefixes)]
        categorized[category] = df[category_features]
        other_features -= set(category_features)

    categorized['OTHER'] = df[list(other_features)]  # Data that doesn't fit any category
    return categorized

def get_category_features(df, category, contain_annotations=False):
    if not contain_annotations:
        return df[[col for col in df.columns if category in col]].copy()
    else:
        return df[[col for col in df.columns if (category in col) or ('arousal'in col or 'valence' in col)]].copy()