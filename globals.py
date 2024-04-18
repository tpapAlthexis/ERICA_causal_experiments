import os

ANNOTATIONS_REGEX = '^((FM|FF)[1-3] ?_[xy])$'

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

PREPROCESSED_PATH = CURRENT_DIR_PATH + '/data/preprocessed'
STANDARDIZED_PATH = CURRENT_DIR_PATH + '/data/standardized'

SEWA_PREPROCESSED_PATH = CURRENT_DIR_PATH + '/data/preprocessed_sewa'
SEWA_STANDARDIZED_PATH = CURRENT_DIR_PATH + '/data/standardized_sewa'

PREPROCESSED_POSTFIX = '_preprocessed'
STANDARDIZED_POSTFIX = '_standardized'

DEFAULT_INPUT_DATA_PATH = 'data/RECOLA-DATA'

SEWA_INPUT_DATA_PATH = 'data/SEWA DB v0.2 - BASIC'

EXPERIMENTAL_DATA_PATH = 'data/experiments'

#enum for RECOLA or SEWA
class DatasetName:
    RECOLA = 1
    SEWA = 2

# Constants
AUDIO = 'audio'
VIDEO = 'video'
ECG = 'ecg'
EDA = 'eda'
OTHER = 'other'
AROUSAL = 'arousal'
VALENCE = 'valence'

Measure_Category_Prefixes = {
    AUDIO: 'ComParE',
    VIDEO: 'VIDEO',
    ECG: 'ECG',
    EDA: 'EDA'
}

def getParticipantStandardizedPath(participant, dataset_name=DatasetName.RECOLA):
    if dataset_name == DatasetName.RECOLA:
        return f'{STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}.csv'
    elif dataset_name == DatasetName.SEWA:
        return f'{SEWA_STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}.csv'
    else:
        return None

def getAnnotationsPath(participant, dataset_name=DatasetName.RECOLA):
    if dataset_name == DatasetName.RECOLA:
        return f'{PREPROCESSED_PATH}/P{str(participant)}{PREPROCESSED_POSTFIX}_annotations_median.csv'
    elif dataset_name == DatasetName.SEWA:
        return f'{SEWA_PREPROCESSED_PATH}/P{str(participant)}{PREPROCESSED_POSTFIX}_annotations_median.csv'
    else:
        return None

def getParticipants(dataset_name=DatasetName.RECOLA):
    if dataset_name == DatasetName.RECOLA:
        file_names = os.listdir(PREPROCESSED_PATH)
    elif dataset_name == DatasetName.SEWA:
        file_names = os.listdir(SEWA_PREPROCESSED_PATH)
    else:
        return None
    participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names]
    return list(set(participants))