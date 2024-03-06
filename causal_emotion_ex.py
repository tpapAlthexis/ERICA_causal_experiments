import independence_causal as ic
import globals as gl
from enum import Enum
import os
from datetime import datetime
import sys

PARTICIPANTS = [16,19,23]#gl.getParticipants()
COMPONENTS_THRESHOLD = ic.COMPONENTS_THRESHOLD
USE_ICA = True
FOLDS = ic.FOLDS
EDGE_CUTOFF = FOLDS / 2
EXPERIMENT_FOLDER_PATH = gl.EXPERIMENTAL_DATA_PATH + '/causal_emotion/' + 'exp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class ExperimentEnum(Enum):
    Setup = 0
    Preproc_logs = 1
    Graphs = 2
    Column_titles = 3

class ExperimentSetup(Enum):
    Participant = 0
    Folds = 1
    Use_ICA = 2
    Components_Threshold = 3
    Edge_Cutoff = 4
    Analysis_Features = 5

def run_causal_emotion_experiment(participant, analysis_features):
    categorized_data, annotation_data = ic.readData(participant)

    exp_dict = {}
    exp_setup = {ExperimentSetup.Participant.name: participant, ExperimentSetup.Folds.name: FOLDS, ExperimentSetup.Use_ICA.name: USE_ICA, ExperimentSetup.Components_Threshold.name: COMPONENTS_THRESHOLD, ExperimentSetup.Edge_Cutoff.name: EDGE_CUTOFF, ExperimentSetup.Analysis_Features.name: analysis_features}
    exp_dict[ExperimentEnum.Setup.name] = exp_setup
    exp_dict[ExperimentEnum.Preproc_logs.name] = ['']
    # Apply preprocessing to the data (PCA, flattening, etc.)
    data_df = ic.preprocess_data(categorized_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA, exp_dict[ExperimentEnum.Preproc_logs.name])

    #select only data containing the features we are interested in
    data_df = data_df[[col for col in data_df.columns if any(feature in col for feature in analysis_features)]]
    exp_dict[ExperimentEnum.Column_titles.name] = data_df.columns

    graphs = ic.run_experiment(data_df, folds=FOLDS)
    exp_dict[ExperimentEnum.Graphs.name] = graphs

    return exp_dict

def run_experiment():

    const_features = [ic.AROUSAL, ic.VALENCE] 
    features = [ic.AUDIO, ic.VIDEO, ic.ECG, ic.EDA, ic.OTHER]

    experiment_res_dict = {}
    path = create_experiment_folder_path()

    for participant in PARTICIPANTS:
        experiment_res_dict[participant] = {}
        for feat in features:
            analysis_features = const_features + [feat]
            experiment_res_dict[participant][feat] = run_causal_emotion_experiment(participant, analysis_features)

            current_experiment = experiment_res_dict[participant][feat]
            edge_histogram = ic.get_edge_histogram(current_experiment[ExperimentEnum.Graphs.name], current_experiment[ExperimentEnum.Column_titles.name], EDGE_CUTOFF)
            ic.create_graph_image(current_experiment[ExperimentEnum.Column_titles.name], edge_histogram, f'{path}/P{participant}_{feat}_graph.png')
    
    return experiment_res_dict

def create_experiment_folder_path():
    if not os.path.exists(EXPERIMENT_FOLDER_PATH):
        os.makedirs(EXPERIMENT_FOLDER_PATH)

    return EXPERIMENT_FOLDER_PATH

if __name__ == "__main__":

    path = create_experiment_folder_path()

    res = run_experiment()

    # for participant in res:
    #     for feat in res[participant]:
    #         print(f'Participant {participant}, feature {feat}')
    #         current_experiment = res[participant][feat]
    #         edge_histogram = ic.get_edge_histogram(current_experiment[ExperimentEnum.Graphs.name], current_experiment[ExperimentEnum.Column_titles.name], EDGE_CUTOFF)
    #         plot_image = ic.get_graph_image(current_experiment[ExperimentEnum.Column_titles.name], edge_histogram= edge_histogram)
    #         #save the image to path
    #         plot_image.savefig(f'{path}/P{participant}_{feat}_graph.png')
    #         print('\n')