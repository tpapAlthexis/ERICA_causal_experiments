import os

ANNOTATIONS_REGEX = '^((FM|FF)[1-3] ?_[xy])$'

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

PREPROCESSED_PATH = CURRENT_DIR_PATH + '/data/preprocessed'
STANDARDIZED_PATH = CURRENT_DIR_PATH + '/data/standardized'

SEWA_PREPROCESSED_PATH = CURRENT_DIR_PATH + '/data/preprocessed_sewa'
SEWA_STANDARDIZED_PATH = CURRENT_DIR_PATH + '/data/standardized_sewa'

TEST_PATH = CURRENT_DIR_PATH + '/PC_tests'

PREPROCESSED_POSTFIX = '_preprocessed'
STANDARDIZED_POSTFIX = '_standardized'

DEFAULT_INPUT_DATA_PATH = 'data/RECOLA-DATA'

SEWA_INPUT_DATA_PATH = 'data/SEWA DB v0.2 - BASIC'

EXPERIMENTAL_DATA_PATH = 'data/experiments'

#enum for RECOLA or SEWA
class Dataset:
    RECOLA = 1
    SEWA = 2

DatasetNames = {
    Dataset.RECOLA: 'RECOLA',
    Dataset.SEWA: 'SEWA'
}

# Constants
AUDIO = 'audio'
VIDEO = 'video'
ECG = 'ecg'
EDA = 'eda'
OTHER = 'other'
AROUSAL = 'arousal'
VALENCE = 'valence'

Measure_Category_Prefixes = {
    AUDIO: ['ComParE', 'audio_'],
    VIDEO: ['VIDEO', 'Face_'],
    ECG: ['ECG'],
    EDA: ['EDA']
}

Selected_audio_features = [
    "ComParE13_LLD_25Hz_F0final_sma_amean",
    "ComParE13_LLD_25Hz_voicingFinalUnclipped_sma_amean",
    "ComParE13_LLD_25Hz_jitterLocal_sma_amean",
    "ComParE13_LLD_25Hz_shimmerLocal_sma_amean",
    "ComParE13_LLD_25Hz_logHNR_sma_amean",
    "ComParE13_LLD_25Hz_pcm_RMSenergy_sma_amean",
    "ComParE13_LLD_25Hz_pcm_zcr_sma_amean",
    "ComParE13_LLD_25Hz_audSpec_Rfilt_sma[0]_amean",
    "ComParE13_LLD_25Hz_pcm_Mag_spectralFlux_sma_amean",
    "ComParE13_LLD_25Hz_pcm_Mag_spectralEntropy_sma_amean",
    "ComParE13_LLD_25Hz_pcm_Mag_spectralCentroid_sma_amean",
    "ComParE13_LLD_25Hz_pcm_Mag_psySharpness_sma_amean",
    "ComParE13_LLD_25Hz_mfcc_sma[1]_amean",
    "ComParE13_LLD_25Hz_mfcc_sma[2]_amean",
    "audio_speech_probability_lstm_vad"
]

Selected_video_features = [
    "VIDEO_40_LLD_AU1",
    "VIDEO_40_LLD_AU4",
    "VIDEO_40_LLD_AU6",
    "VIDEO_40_LLD_AU12",
    "VIDEO_40_LLD_AU15",
    "VIDEO_40_LLD_AU20",
    "VIDEO_40_LLD_AU25",
    "VIDEO_40_LLD_Yaw",
    "VIDEO_40_LLD_Pitch",
    "VIDEO_40_LLD_Roll",
    "VIDEO_40_LLD_Opt_mean",
    "VIDEO_40_LLD_AU12_delta",
    "VIDEO_40_LLD_AU4_delta",
    "VIDEO_40_LLD_AU6_delta",
    "Face_detection_probability"
]

def getParticipantStandardizedPath(participant, dataset_name=Dataset.RECOLA):
    if dataset_name == Dataset.RECOLA:
        return f'{STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}.csv'
    elif dataset_name == Dataset.SEWA:
        return f'{SEWA_STANDARDIZED_PATH}/P{str(participant)}{STANDARDIZED_POSTFIX}.csv'
    else:
        return None

def getAnnotationsPath(participant, dataset_name=Dataset.RECOLA):
    if dataset_name == Dataset.RECOLA:
        return f'{PREPROCESSED_PATH}/P{str(participant)}{PREPROCESSED_POSTFIX}_annotations_median.csv'
    elif dataset_name == Dataset.SEWA:
        return f'{SEWA_PREPROCESSED_PATH}/P{str(participant)}_annotations.csv'
    else:
        return None

def getParticipants(dataset_name=Dataset.RECOLA):
    if dataset_name == Dataset.RECOLA:
        file_names = os.listdir(PREPROCESSED_PATH)
    elif dataset_name == Dataset.SEWA:
        file_names = os.listdir(SEWA_STANDARDIZED_PATH)
    else:
        return None
    participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names]
    return list(set(participants))

def getAnnotationStandardizationCompatibility(dataset_name=Dataset.RECOLA):
    if dataset_name == Dataset.RECOLA:
        file_names_ann = os.listdir(PREPROCESSED_PATH)
        file_names_std = os.listdir(STANDARDIZED_PATH)
    elif dataset_name == Dataset.SEWA:
        file_names_ann = os.listdir(SEWA_PREPROCESSED_PATH)
        file_names_std = os.listdir(SEWA_STANDARDIZED_PATH)
   
    participants_ann = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names_ann if file_name.endswith('_annotations_median.csv')]
    standardized_files = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names_std]
    
    print('Participants with annotations:', participants_ann)
    print('Standardized files:', standardized_files)
    print('Participants with annotations but no standardized files:', list(set(participants_ann) - set(standardized_files)))
    print('Standardized files but no annotations:', list(set(standardized_files) - set(participants_ann)))
    return set(participants_ann) == set(standardized_files)
    