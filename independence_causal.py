import globals as gl
import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils import GraphUtils

import networkx as nx
import matplotlib.pyplot as plt


PARTICIPANT = 23
FOLDS = 10
expertiment_path = gl.EXPERIMENTAL_DATA_PATH + f'/independence/_P{PARTICIPANT}'
if not os.path.exists(expertiment_path):
    os.makedirs(expertiment_path)

def clear_data(data_df):
    time_col = [col for col in data_df.columns if 'time' in col]
    if time_col:
        data_df = data_df.drop(columns=time_col)
    time_col = [col for col in data_df.columns if 'Unnamed: 0' in col]
    if time_col:
        data_df = data_df.drop(columns=time_col)
    return data_df

def readData(participant):
    data_df = clear_data(pd.read_csv(gl.getParticipantTrainingPath(participant)))
    annotations_df = clear_data(pd.read_csv(gl.getAnnotationsPath(participant)))

    return data_df, annotations_df

def preprocess_data(data_df, annotations_df, components_threshold=50):
    categorized_data = categorize_columns(data_df)
    pca_data = apply_pca_to_categories(categorized_data, 0.95, components_threshold)

    #print annotations first column title
    if annotations_df.columns.size > 2:
        print("Incopatible annotations size")
        sys.exit(0)

    #concut pca_data , set as key the column's title and value the annotation data that follow
    pca_data[annotations_df.columns[0]] = annotations_df[annotations_df.columns[0]]
    pca_data[annotations_df.columns[1]] = annotations_df[annotations_df.columns[1]]

    flattened_data = {}
    keys_to_remove = []
    for category, data in pca_data.items():
        if isinstance(data, np.ndarray) and data.ndim == 2:  # Check if data is a 2D numpy array
            for i in range(data.shape[1]):
                new_key = f"{category}_pc_{i+1}"
                flattened_data[new_key] = data[:, i]
            keys_to_remove.append(category)

    # Remove the original multi-dimensional data from pca_data
    for key in keys_to_remove:
        pca_data.pop(key)

    # Merge the flattened data with the original pca_data
    pca_data.update(flattened_data)

    # Remove columns with more than 50% of missing values
    data_df = data_df.dropna(thresh=data_df.shape[0] * 0.5, axis=1)

    # Remove rows with missing values
    data_df = data_df.dropna()

    return pd.DataFrame(pca_data)

# Function to categorize columns
def categorize_columns(df):
    audio_features = [col for col in df.columns if col.startswith('ComParE')]
    video_features = [col for col in df.columns if col.startswith('VIDEO')]
    ecg_features = [col for col in df.columns if col.startswith('ECG')]
    eda_features = [col for col in df.columns if col.startswith('EDA')]
    other_features = [col for col in df.columns if col not in audio_features + video_features + ecg_features + eda_features]

    return {
        'audio': df[audio_features], #voice features
        'video': df[video_features], #facial features
        'ecg': df[ecg_features], #heart features, physiology
        'eda': df[eda_features], #skin features, physiology
        'other': df[other_features] #other features
    }

# Function to apply PCA to each category and retain components explaining 95% variance
def apply_pca_to_categories(categorized_data, variance_threshold=0.95, components_threshold=50):
    pca_results = {}
    print("-------------------")
    print("PCA results")
    for category, data in categorized_data.items():
        pca = PCA(n_components=variance_threshold, svd_solver='full')  
        components = pca.fit_transform(data)

        if components.shape[1] > components_threshold:
            pca = PCA(n_components=components_threshold, svd_solver='full') 
            components = pca.fit_transform(data)

        pca_results[category] = components
        
        print(f"{category} - Original shape: {data.shape}, Reduced shape: {components.shape}, Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")

    print("-------------------" )
    return pca_results

def save_graph(graph, labels, path):
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    tmp_png = pyd.create_png(f="png")
    # Save png to path
    with open(path, 'wb') as f:
        f.write(tmp_png)

def plot_histogram_edges(column_titles, edge_histogram):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for title in column_titles:
        G.add_node(title)

    # Add edges to the graph
    for edge, weight in edge_histogram.items():
        edge_nodes = edge.split('-->')
        edge_nodes = [node.strip() for node in edge_nodes]
        G.add_edge(edge_nodes[0], edge_nodes[1], weight=weight)

    # Get edge weights in the order they are stored in G
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Draw the graph
    pos = nx.spring_layout(G)  # Compute position of nodes for the layout of G
    nx.draw_networkx_nodes(G, pos, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='blue', alpha=0.5,
                           arrowstyle='-|>', arrowsize=20,
                           connectionstyle='arc3, rad = 0.1')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

def run_experiment(data_df):
    fold_count = 1
    graphs = []

    for train_index, test_index in kf.split(data_df):
        # Split data into train and test
        train_data = data_df.iloc[train_index].values
        test_data = data_df.iloc[test_index].values

        try:
            # Run PC algorithm on train data
            cg_train = pc(train_data, 0.05, fisherz)
            #cg_test = pc(test_data, 0.05, fisherz)
            
            gs = cg_train.G.__str__()
            print(f'Graph string:{gs}')
            
            # if '>' in gs:
            #     save_graph(cg_train.G, column_titles, f'{expertiment_path}/graph_fold_{fold_count}.png')
                #cg_train.draw_pydot_graph(labels=column_titles)

            graphs.append(cg_train)
            fold_count += 1

        except np.linalg.LinAlgError:
            print(f'Error in fold {fold_count}')
            fold_count += 1
            continue
    return graphs

def get_edge_histogram(graphs, column_titles, edge_cutoff=5):
    #create a map from 'X1' to 'Xn' to the column title
    column_title_map = {}
    for i in range(len(column_titles)):
        column_title_map[f'X{i+1}'] = column_titles[i]
        
    #for each graph
    fold_num = 0
    edge_histogram = {}
    for graph in graphs:
        set_edges = set()
        graph = graphs[fold_num].G

        graph_nodes = graph.get_node_names()
        for node1 in graph_nodes:
            for node2 in graph_nodes:
                if node1 != node2:
                    edge = graph.get_edge(graph.get_node(node1), graph.get_node(node2))
                    if '>' in edge.__str__():
                        set_edges.add(edge.__str__())
        
        #for each edge string replace 'X1' to 'Xn' with the column title
        for i in range(len(column_titles)):
            set_edges = [edge.replace(f'X{i+1}', column_title_map[f'X{i+1}']) for edge in set_edges]

        for edge in set_edges:
            if edge in edge_histogram:
                edge_histogram[edge] += 1
            else:
                edge_histogram[edge] = 1

        fold_num += 1

    #sort histogram according to edge count
    edge_histogram = {k: v for k, v in sorted(edge_histogram.items(), key=lambda item: item[1], reverse=True)}

    #remove edges with count less than edge_cutoff
    edge_histogram = {k: v for k, v in edge_histogram.items() if v >= edge_cutoff}

    return edge_histogram

def create_new_graph(column_titles, edge_histogram):
    final_nodes = []
    for title in column_titles:
        final_nodes.append(GraphNode(title))

    complete_graph = GeneralGraph(nodes=final_nodes)
    for edge in edge_histogram:
        edge_nodes = edge.split('-->')
        #remove ' ' for each node
        edge_nodes = [node.strip() for node in edge_nodes]
        print(f'Edge nodes: {edge_nodes}')

        complete_graph.add_directed_edge(complete_graph.get_node(edge_nodes[0]), complete_graph.get_node(edge_nodes[1]))

    print(f'Graph: {complete_graph}')

if __name__ == "__main__":

    # Categorize the columns of your dataframe
    categorized_data, annotation_data = readData(PARTICIPANT)

    # Apply preprocessing to the data (PCA, flattening, etc.)
    data_df = preprocess_data(categorized_data, annotation_data, components_threshold=3)

    kf = KFold(n_splits=FOLDS)

    #select only data containing 'audio' or 'arousal' or 'valence' category (check if 'audio' is present in the column's title)
    data_df = data_df[[col for col in data_df.columns if 'audio' in col or 'arousal' in col or 'valence' in col]]
    column_titles = data_df.columns

    graphs = run_experiment(data_df)
    edge_histogram = get_edge_histogram(graphs, column_titles, FOLDS/2)
    print(f'Edge histogram: {edge_histogram}')

    plot_histogram_edges(column_titles, edge_histogram)

