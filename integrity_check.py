import os
import pandas as pd

import globals as gl

P_ANN_NO_STD = 'Participants with annotations but no standardized files:'
P_STD_NO_ANN = 'Participants with standardized files but no annotations:'

def is_ready_for_experiment(dataset):
    preproc_files = os.listdir(gl.PREPROCESSED_PATH) if dataset == gl.Dataset.RECOLA else os.listdir(gl.SEWA_PREPROCESSED_PATH)
    
    # get all files with annotations
    if dataset == gl.Dataset.RECOLA:
        annotation_paths = [f for f in preproc_files if f.endswith('_annotations_median.csv')]
    elif dataset == gl.Dataset.SEWA:
        annotation_paths = [f for f in preproc_files if f.endswith('annotations.csv')]

    #get annotation participants
    annotation_participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in annotation_paths]

    #get standardized participants
    standard_path = os.listdir(gl.STANDARDIZED_PATH) if dataset == gl.Dataset.RECOLA else os.listdir(gl.SEWA_STANDARDIZED_PATH)
    standard_p_paths = [f for f in standard_path if f.endswith(gl.STANDARDIZED_POSTFIX + '.csv')]

    standardized_participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in standard_p_paths]

    #sort annotation participant and standardized participants according to their participant number
    annotation_participants.sort()
    standardized_participants.sort()

    p_to_avoid = None
    state = True

    #check if all participants have standardized files
    if annotation_participants != standardized_participants:
        ann_n_std_p = list(set(annotation_participants) - set(standardized_participants))
        std_n_ann_p = list(set(standardized_participants) - set(annotation_participants))

        print(P_ANN_NO_STD, sorted(ann_n_std_p))
        print(P_STD_NO_ANN, sorted(std_n_ann_p))
        p_to_avoid = set(ann_n_std_p + std_n_ann_p)
        state = False
    
    print('All participants have standardized files matching the annotations')

    #check if all annotation/participant files have the same number of rows
    for participant in annotation_participants:
        if p_to_avoid and participant in p_to_avoid:
            continue

        annotation_path = gl.getAnnotationsPath(participant, dataset)
        standardized_path = gl.getParticipantStandardizedPath(participant, dataset)

        if not os.path.exists(annotation_path) or not os.path.exists(standardized_path):
            print(f'Participant {participant} has no annotation or standardized file')
            state = False
            p_to_avoid.add(participant)
            continue

        annotations = pd.read_csv(annotation_path)
        standardized = pd.read_csv(standardized_path)

        if len(annotations) != len(standardized):
            print(f'Participant {participant} has {len(annotations)} annotations and {len(standardized)} standardized rows')
            state = False
            p_to_avoid.add(participant)
    
    if p_to_avoid:
        print(f"Participants to avoid: {sorted(list(p_to_avoid))}")
    
    return list([] if not p_to_avoid else p_to_avoid)