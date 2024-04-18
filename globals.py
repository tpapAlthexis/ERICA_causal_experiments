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

def getParticipantStandardizedPath(participant):
    return f'{STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}.csv'

def getAnnotationsPath(participant):
    return f'{PREPROCESSED_PATH}/P{str(participant)}{PREPROCESSED_POSTFIX}_annotations_median.csv'

def getAnnotationsPathStd(participant):
    return f'{STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}_annotations_median.csv'

def getParticipants():
    file_names = os.listdir(PREPROCESSED_PATH)
    participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names]
    return list(set(participants))