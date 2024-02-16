import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.SHD import SHD
import matplotlib.pyplot as plt
from enum import Enum
import sys
import os

import globals as gl

class EmotionEnum(Enum):
    arousal = 0
    valence = 1

PARTICIPANT = 16
FOLDS = 10
EMOTION = EmotionEnum.arousal.value

assert FOLDS > 0

def calculate_shd_scores(G1, G2, normalization_factor=1.0):
    shd = SHD(G1, G2).get_shd()
    
    real_edges = G1.get_num_edges()
    estimated_edges = G2.get_num_edges()
    
    normalized_shd = shd * normalization_factor
    
    real_nodes = G1.get_num_nodes()
    estimated_nodes = G2.get_num_nodes()
    
    print(f'Shd: {shd}, real_edg: {real_edges}, est_edges: {estimated_edges}')
    
    return real_nodes, estimated_nodes, real_edges, estimated_edges, shd, normalized_shd

def trainProcedure(participant=PARTICIPANT, folds=FOLDS, emotion=EMOTION):
    # Load data from csv file to DataFrame
    data_df = pd.read_csv(gl.getParticipantTrainingPath(participant))
    annotations_df = pd.read_csv(gl.getAnnotationsPath(participant))
    
    #find and delete 'time' column from both dataframes
    time_col = [col for col in data_df.columns if 'time' in col]
    if time_col:
        data_df = data_df.drop(columns=time_col)
    
    time_col = [col for col in annotations_df.columns if 'time' in col]
    if time_col:
        annotations_df = annotations_df.drop(columns=time_col)

    print(f'Data shape: {data_df.shape}')
    print(f'Annotations shape: {annotations_df.shape}')
    print("-------------------")

    kf = KFold(n_splits=folds)
    columns_to_avoid = ['frame_path', 'Unnamed: 0'] # just in case they are present in the csv file
    
    graph_data = {}

    # Iterate over each measure
    for column_name in data_df.columns:
        if column_name in columns_to_avoid:
            continue

        print(f"Processing column {column_name}...")

        # Select only the current measure and the arousal values
        selected_data = pd.concat([data_df[[column_name]], annotations_df.iloc[:, emotion]], axis=1)

        #selected_data = pd.concat([data_df[[column_name]], annotations_df], axis=1)
        
        fold_count = 1
        graph_data[column_name] = []
        
        for train_index, test_index in kf.split(selected_data):
            # Split data into train and test
            train_data = selected_data.iloc[train_index].values
            test_data = selected_data.iloc[test_index].values

            try:
                # Run PC algorithm on train data
                cg_train = pc(train_data, 0.05, fisherz)
                #cg_test = pc(test_data, 0.05, fisherz)
                
                gs = cg_train.G.__str__()
                print(f'Graph string:{gs}')
                
                if '>' in gs:
                    cg_train.draw_pydot_graph(labels=[column_name, 'arousal'])
                
                #append train graph to graph_data with key the column name and second key the current fold
                graph_data[column_name].append([fold_count, cg_train])
                fold_count += 1

            except np.linalg.LinAlgError:
                print(f"Could not compute PC for column {column_name} due to singular matrix error.")
                continue
            except Exception as e:
                print(f"An error occurred while processing column {column_name}.")
                print(e)
                continue
            
    return graph_data

def get_ranked_edge_data(graph_data):
    
    ranked_measures = []
    
    for measure in graph_data:
        edge_count = 0
        for data in graph_data[measure]:
            edge_count += data[1].G.get_num_edges()
        ranked_measures.append((measure, edge_count))
    
    ranked_measures.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_measures

def get_data_stats(ranked_measures):
    stats = {}
    
    # for each edge count in the ranked measures count measures with the same edge count
    for measure in ranked_measures:
        edge_count = measure[1]
        if edge_count not in stats:
            stats[edge_count] = 0
        stats[edge_count] += 1
    
    return stats

def saveExpData(edges_per_measure, edge_count_histogram, participant=PARTICIPANT, folds=FOLDS, emotionName=EmotionEnum.arousal.name):
    if not os.path.exists(gl.EXPERIMENTAL_DATA_PATH):
        os.makedirs(gl.EXPERIMENTAL_DATA_PATH)
    experiment_folder_path = gl.EXPERIMENTAL_DATA_PATH + f'/P{participant}_f{folds}_{emotionName}'
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
        
    with open(experiment_folder_path + '/experiment_data.json', 'w') as f:
        json.dump({'edges_per_measure': edges_per_measure, 'edge_count_histogram': edge_count_histogram}, f, indent=4)
        
    #clear the plot
    plt.clf()
    plt.bar(edge_count_histogram.keys(), edge_count_histogram.values())
    plt.title(f'participant={participant}, folds={folds}, emotion={emotionName}')  # Set the title
    plt.xlabel('Number of Edges')  # Set the x-axis label
    plt.ylabel('Number of Measures')  # Set the y-axis label
    # plt.show()  # Display the plot
    
    #save athe plot
    plt.savefig(experiment_folder_path + '/edge_count_histogram.png')

if __name__ == "__main__":

    if len(sys.argv) > 1:
        for arg in sys.argv:
            if 'participant' in arg:
                PARTICIPANT = int(arg.split('=')[1])
            elif 'folds' in arg:
                FOLDS = int(arg.split('=')[1])
            elif 'emotion' in arg:
                EMOTION = int(arg.split('=')[1])
                
    print("-------------------")
    print(f'Experiment parameters: participant={PARTICIPANT}, folds={FOLDS}, emotion={EmotionEnum(EMOTION).name}')
    print("-------------------")
                
    graph_data = trainProcedure()
    edges_per_measure = get_ranked_edge_data(graph_data)
    edge_count_histogram = get_data_stats(edges_per_measure)
    
    for stat in edge_count_histogram:
        print(f'Edges: {stat} - Measures: {edge_count_histogram[stat]}, percentage: {edge_count_histogram[stat] / len(edges_per_measure) * 100:.2f}%')
    
    saveExpData(edges_per_measure, edge_count_histogram, PARTICIPANT, FOLDS, EmotionEnum(EMOTION).name)
    
    print("-------------------")
    print("Finished processing all columns.")
    print("-------------------")