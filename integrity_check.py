import os
import pandas as pd

import globals as gl

def is_ready_for_experiment(dataset):
    preproc_files = os.listdir(gl.PREPROCESSED_PATH) if dataset == gl.DatasetName.RECOLA else os.listdir(gl.SEWA_PREPROCESSED_PATH)
    
    # get all files with annotations
    if dataset == gl.DatasetName.RECOLA:
        annotation_paths = [f for f in preproc_files if f.endswith('_annotations_median.csv')]
    elif dataset == gl.DatasetName.SEWA:
        annotation_paths = [f for f in preproc_files if f.endswith('annotations.csv')]

    #get annotation participants
    annotation_participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in annotation_paths]

    #get standardized participants
    standard_path = os.listdir(gl.STANDARDIZED_PATH) if dataset == gl.DatasetName.RECOLA else os.listdir(gl.SEWA_STANDARDIZED_PATH)
    standard_p_paths = [f for f in standard_path if f.endswith(gl.STANDARDIZED_POSTFIX + '.csv')]

    standardized_participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in standard_p_paths]

    #sort annotation participant and standardized participants according to their participant number
    annotation_participants.sort()
    standardized_participants.sort()

    #check if all participants have standardized files
    if annotation_participants != standardized_participants:
        print('Participants with annotations but no standardized files:', list(set(annotation_participants) - set(standardized_participants)))
        print('Participants with standardized files but no annotations:', list(set(standardized_participants) - set(annotation_participants)))
        return False
    
    print('All participants have standardized files matching the annotations')

    #check if all annotation/participant files have the same number of rows
    for participant in annotation_participants:
        annotation_path = gl.getAnnotationsPath(participant, dataset)
        standardized_path = gl.getParticipantStandardizedPath(participant, dataset)

        annotations = pd.read_csv(annotation_path)
        standardized = pd.read_csv(standardized_path)

        if len(annotations) != len(standardized):
            print(f'Participant {participant} has {len(annotations)} annotations and {len(standardized)} standardized rows')
            return False
    
    return True