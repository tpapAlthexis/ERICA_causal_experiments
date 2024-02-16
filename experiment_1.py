#in this experiment I will run the measure_emotion_link.py for each participant, emotion and 2,5,10 folds.
#I will calculate the mean and standard deviation of the edge count for each measure.
#I will caluclate the overall edge count for each measure for all participants with each emotion and 2,5,10 folds.

import os
import sys
import json
import pandas as pd
import numpy as np
import measure_emotion_link as mel
import globals as gl
from measure_emotion_link import EmotionEnum

def getParticipants():
    file_names = os.listdir(gl.PREPROCESSED_PATH)
    participants = [int(file_name.split('P')[1].split('_')[0]) for file_name in file_names]
    return list(set(participants))

FOLDS = [2, 6, 10] 
PARTICIPANTS = getParticipants()

def runExperiment(participant, emotion, folds):
    sys.argv = ['measure_emotion_link.py', f'participant={participant}', f'folds={folds}', f'emotion={emotion.value}']
    print(f"Running experiment for participant {participant}, emotion {emotion}, folds {folds}")
    
    sys.stdout = open(os.devnull, 'w')
    #mute general output 
    graph_data = mel.trainProcedure(participant, folds, emotion.value)
    sys.stdout = sys.__stdout__
    
    edges_per_measure = mel.get_ranked_edge_data(graph_data)
    edge_count_histogram = mel.get_data_stats(edges_per_measure)
    mel.saveExpData(edges_per_measure, edge_count_histogram, participant, folds, emotion.name)
    
    return edges_per_measure

def getEdgeCountMean(edges_per_measure):
    #calculate the mean of the edge count for each measure.
    edge_count_mean = {}
    for measure in edges_per_measure:
        edge_count = 0
        for data in edges_per_measure[measure]:
            edge_count += data[1].G.get_num_edges()
        edge_count_mean[measure] = edge_count / len(edges_per_measure[measure])
    return edge_count_mean

def getEdgeCountStd(edges_per_measure, edge_count_mean):
    #calculate the standard deviation of the edge count for each measure.
    edge_count_std = {}
    for measure in edges_per_measure:
        edge_count = 0
        for data in edges_per_measure[measure]:
            edge_count += (data[1].G.get_num_edges() - edge_count_mean[measure])**2
        edge_count_std[measure] = (edge_count / len(edges_per_measure[measure]))**0.5
    return edge_count_std

def getOverallEdgeCount(edges_per_measure):
    #caluclate the overall edge count for each measure for all participants with each emotion and 2,5,10 folds.
    overall_edge_count = {}
    for measure in edges_per_measure:
        edge_count = 0
        for data in edges_per_measure[measure]:
            edge_count += data[1].G.get_num_edges()
        overall_edge_count[measure] = edge_count
    return overall_edge_count

def saveExperimentData(participant, emotion, folds, edge_count_mean, edge_count_std, overall_edge_count):
    print(f"Saving experiment data for participant {participant}, emotion {emotion}, folds {folds}")
    if not os.path.exists(gl.EXPERIMENTAL_DATA_PATH):
        os.makedirs(gl.EXPERIMENTAL_DATA_PATH)
    experiment_folder_path = gl.EXPERIMENTAL_DATA_PATH + f'/P{participant}_f{folds}_e{emotion}'
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    with open(experiment_folder_path + '/experiment_data.json', 'w') as f:
        json.dump({'edge_count_mean': edge_count_mean, 'edge_count_std': edge_count_std, 'overall_edge_count': overall_edge_count}, f)

def main():
    for participant in PARTICIPANTS:
        for emotion in EmotionEnum:
            for folds in FOLDS:
                print(f"Starting experiment for participant {participant}, emotion {emotion}, folds {folds}")
                runExperiment(participant, emotion, folds)
                print(f"Finished experiment for participant {participant}, emotion {emotion}, folds {folds}")
                
if __name__ == "__main__":
    print("Starting the experiment.")
    main()
    print("Experiment finished.")

