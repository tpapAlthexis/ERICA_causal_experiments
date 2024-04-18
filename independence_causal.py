import io
import globals as gl
import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import json
import warnings

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils import GraphUtils

LOG_SEPARATOR = '$'

DATASET_NAME = gl.DatasetName.SEWA

# Parameters
PARTICIPANT = 1 # participant number, ex participants: 16, 19, 21, 23, 25, 26, 28  
FOLDS = 5 # number of folds
COMPONENTS_THRESHOLD = 15 # number of PCA compontents. If expl. variance is lower than 95% and PCs are more, then reduct to current number
EDGE_CUTOFF = FOLDS / 2 # number of edges to be included in the histogram. If edge count is less than this number, then remove it
USE_ICA = True # use ICA instead of PCA
ANALYSIS_FEATURES = [gl.AUDIO, gl.AROUSAL, gl.VALENCE]

EXPERIMENT_SETUP = f'P{PARTICIPANT}_F{FOLDS}_C{COMPONENTS_THRESHOLD}_E{EDGE_CUTOFF}_ICA{USE_ICA}'
EXPERIMENT_ATTR = ''

expertiment_path = gl.EXPERIMENTAL_DATA_PATH + f'/independence/_P{PARTICIPANT}'
if not os.path.exists(expertiment_path):
    os.makedirs(expertiment_path)

def clear_data(data_df):
    drop_col = [col for col in data_df.columns if 'time' in col]
    if drop_col:
        data_df = data_df.drop(columns=drop_col)
    frame_col = [col for col in data_df.columns if 'frame_range' in col]
    if frame_col:
        data_df = data_df.drop(columns=frame_col)
    drop_col = [col for col in data_df.columns if 'Unnamed: 0' in col]
    if drop_col:
        data_df = data_df.drop(columns=drop_col)
    return data_df

def readData(participant, dataset=DATASET_NAME):
    data_df = clear_data(pd.read_csv(gl.getParticipantStandardizedPath(participant, dataset)))
    annotations_df = clear_data(pd.read_csv(gl.getAnnotationsPath(participant, dataset)))

    return data_df, annotations_df

def readDataAll_p(measures_list, dataset=DATASET_NAME):
    participants = gl.getParticipants()

    data = {}
    annotations = []

    #read data and annotations for each measure and for all participants
    for measure in measures_list:
        data_measure = []
        for participant in participants:
            data_measure_df = clear_data(pd.read_csv(gl.getParticipantStandardizedPath(participant, dataset)))

            #keep only the features we are interested in
            if not measure == gl.OTHER:
                data_measure_df = data_measure_df[[col for col in data_measure_df.columns if col.startswith(gl.Measure_Category_Prefixes[measure])]]
            else:
                data_measure_df = data_measure_df[[col for col in data_measure_df.columns if not any(prefix in col for prefix in gl.Measure_Category_Prefixes.values())]]
    
            data_measure.append(data_measure_df)
        
        data[measure] = pd.concat(data_measure)

    for participant in participants:
        annotations_df = clear_data(pd.read_csv(gl.getAnnotationsPath(participant, dataset)))
        annotations.append(annotations_df)

    data_df = pd.concat(data.values(), axis=1)
    annotations_df = pd.concat(annotations)
           
    return data_df, annotations_df

def validate_categorized_data(categorized_data, min_samples=10, min_features=2):
    """
    Validates the categorized data for PCA/ICA analysis.
    
    Parameters:
    - categorized_data: dict, with categories as keys and pandas DataFrame/2D numpy array as values.
    - min_samples: int, minimum number of samples required to proceed with the analysis.
    - min_features: int, minimum number of features required to proceed with the analysis.
    
    Returns:
    - valid_data: dict, containing only the categories with valid data for analysis.
    - invalid_categories: list, categories that did not meet the validation criteria.
    """
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
            print(f"validate_categorized_data: Data for category '{category}' contains NaN values. Consider preprocessing to handle NaNs.")
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

def preprocess_data(data_df, annotations_df, components_threshold=50, use_ica=USE_ICA, proc_logs=[''], analysis_features=None):
    categorized_data = categorize_columns(data_df)
    #keep only the features we are interested in
    if analysis_features:
        categorized_data = {key: value for key, value in categorized_data.items() if key in analysis_features}

    if use_ica:
        component_data = apply_ica_to_categories(categorized_data, 0.95, components_threshold, proc_logs)
    else:
        component_data = apply_pca_to_categories(categorized_data, 0.95, components_threshold, proc_logs)

    if component_data is None:
        print("preprocess_data: No component data. This is probably due to convergence failure or incopatible data size. Exiting...")
        sys.exit(0)

    if annotations_df.columns.size != 2:
        print("preprocess_data: Incopatible annotations size")
        sys.exit(0)

    #concut component_data , set as key the column's title and value the annotation data that follow
    component_data[annotations_df.columns[0]] = annotations_df[annotations_df.columns[0]]
    component_data[annotations_df.columns[1]] = annotations_df[annotations_df.columns[1]]

    flattened_data = {}
    keys_to_remove = []
    for category, data in component_data.items():
        if isinstance(data, np.ndarray) and data.ndim == 2:  # Check if data is a 2D numpy array
            for i in range(data.shape[1]):
                new_key = f"{category}_comp_{i+1}"
                flattened_data[new_key] = data[:, i]
            keys_to_remove.append(category)

    # Remove the original multi-dimensional data from component_data
    for key in keys_to_remove:
        component_data.pop(key)

    # Merge the flattened data with the original component_data
    component_data.update(flattened_data)

    return pd.DataFrame(component_data)

# Function to categorize columns
def categorize_columns(df):
    audio_features = [col for col in df.columns if col.startswith(gl.Measure_Category_Prefixes[gl.AUDIO])]
    video_features = [col for col in df.columns if col.startswith(gl.Measure_Category_Prefixes[gl.VIDEO])]
    ecg_features = [col for col in df.columns if col.startswith(gl.Measure_Category_Prefixes[gl.ECG])]
    eda_features = [col for col in df.columns if col.startswith(gl.Measure_Category_Prefixes[gl.EDA])]
    other_features = [col for col in df.columns if col not in audio_features + video_features + ecg_features + eda_features]

    return {
        gl.AUDIO: df[audio_features], #voice features
        gl.VIDEO: df[video_features], #facial features
        gl.ECG: df[ecg_features], #heart features, physiology
        gl.EDA: df[eda_features], #skin features, physiology
        gl.OTHER: df[other_features] #other features
    }

def get_category_features(df, category, contain_annotations=False):
    if not contain_annotations:
        return df[[col for col in df.columns if category in col]].copy()
    else:
        return df[[col for col in df.columns if (category in col) or ('arousal'in col or 'valence' in col)]].copy()

# Function to apply PCA to each category and retain components explaining 95% variance
def apply_pca_to_categories(categorized_data, variance_threshold=0.95, components_threshold=50, proc_logs=['']):
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
        
        pca_log = f"PCA{category} - Original shape: {data.shape}, Explained variance: {np.sum(pca.explained_variance_ratio_):.2f} for {components.shape[1]}, Reduced to components number: {components.shape[1]}"
        proc_logs[category] += f"PCA setup:\n{pca_log}"
        print(proc_logs[category])

    print("-------------------" )
    return pca_results

def apply_ica_to_categories(categorized_data, variance_threshold=0.95, components_threshold=50, proc_logs=['']):
    pca_results = {}
    ica_results = {}
    valid_data, invalid_categories = validate_categorized_data(categorized_data)
    if invalid_categories:
        print("Some categories were invalid and will be skipped:", invalid_categories)

    print("-------------------")
    print("ICA results")
    for category, data in valid_data.items():
        pca = PCA(n_components=variance_threshold, svd_solver='full')  
        pca_components = pca.fit_transform(data)
        num_of_components = components_threshold if pca_components.shape[1] > components_threshold else pca_components.shape[1]

        pca_results[0] = pca_components
        proc_logs[0] = f"Pre-analysis: PCA{LOG_SEPARATOR}Measure category: {category}{LOG_SEPARATOR}Original shape: {data.shape}{LOG_SEPARATOR}Explained variance: {100.00 * np.sum(pca.explained_variance_ratio_):.2f}% for {pca_components.shape[1]} components{LOG_SEPARATOR}Reduced to components number: {num_of_components}"
        print(proc_logs[0])
        # Keep log of explained variance for the number of components used
        if pca_components.shape[1] > components_threshold:
            pca = PCA(n_components=components_threshold, svd_solver='full')
            pca_components = pca.fit_transform(data)
            expl_log = f"Explained variance for {pca_components.shape[1]} components: {np.sum(pca.explained_variance_ratio_):.2f}"
            proc_logs[0] += f"{LOG_SEPARATOR}{expl_log}"
            print(expl_log)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ica = FastICA(n_components=num_of_components, tol=0.1, random_state=0, max_iter=1000)
                ica_components = ica.fit_transform(data)
                if len(w) > 0 and "did not converge" in str(w[-1].message):
                    raise UserWarning("ICA failed to converge")
        except UserWarning as e:
            print(f'apply_ica_to_categories: converge failure - {str(e)}')
            proc_logs[0] += f"{LOG_SEPARATOR}ICA failed to converge for category: {category}. Category shape: {data.shape}, Number of components: {num_of_components}. Category will not contained to the analysis"
            continue

        ica_results[category] = ica_components
        ica_log = f"{LOG_SEPARATOR}Components exported with: ICA{LOG_SEPARATOR}Number of iterations: {ica.n_iter_} from max iterations: 1000 ~ {"Converged" if ica.n_iter_ / 1000 < 1.00 else "Not converged!"}"
        proc_logs[0] += f"\n{ica_log}"
        print(ica_log)
    print("-------------------" )

    return ica_results

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
                           arrowstyle='->', arrowsize=45,
                           connectionstyle='arc3, rad = 0.2')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

def run_experiment(data_df, folds=FOLDS, node_names=None, cv=None, groups=None):

    if cv is None:
        cv = KFold(n_splits=folds, shuffle=False, random_state=None)
    elif isinstance(cv, GroupKFold):
        if groups is None:
            print("run_experiment: GroupKFold is specified but groups are not provided.")
            return
    else:
        if groups is None:
            print("run_experiment: Cv is specified but groups are not provided.")

    fold_count = 1
    graphs = []

    for train_index, test_index in cv.split(data_df, groups=groups):
        train_data = data_df.iloc[train_index].values

        # print indices of the columns that are used in the train data
        print(f'Fold {fold_count} - Train data columns: {train_index}. Test data columns: {test_index}')
        print(f'Fold {fold_count} - Train data shape: {train_data.shape}. Test data shape: {data_df.iloc[test_index].values.shape}')

        try:
            # Run PC algorithm on train data
            cg_train = pc(data=train_data, alpha=0.05, indep_test=fisherz, node_names=node_names)
            
            gs = cg_train.G.__str__()
            print(f'Graph string:{gs}')

            graphs.append(cg_train)
            fold_count += 1

        except np.linalg.LinAlgError:
            print(f'Error in fold {fold_count}')
            fold_count += 1
            continue
    return graphs

def get_edge_histogram(graphs, edge_cutoff=5):
    if not graphs:
        print("get_edge_histogram: No graphs provided.")
        return
    #for each graph
    fold_num = 0
    edge_histogram = {}
    for graph in graphs:
        set_edges = set()
        graph_general = graph.G

        graph_nodes = graph_general.get_node_names()
        for node1 in graph_nodes:
            for node2 in graph_nodes:
                if node1 != node2:
                    edge = graph_general.get_edge(graph_general.get_node(node1), graph_general.get_node(node2))
                    if '>' in edge.__str__():
                        set_edges.add(edge.__str__())

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

def get_nodes_from_histogram(edge_histogram):
    nodes = set()
    for edge in edge_histogram:
        edge_nodes = edge.split('-->')
        edge_nodes = [node.strip() for node in edge_nodes]
        nodes.add(edge_nodes[0])
        nodes.add(edge_nodes[1])

    return [GraphNode(node) for node in nodes]

def create_new_graph(edge_histogram):
    try:
        final_nodes = get_nodes_from_histogram(edge_histogram)

        complete_graph = GeneralGraph(nodes=final_nodes)
        for edge in edge_histogram:
            edge_nodes = edge.split('-->')
            edge_nodes = [node.strip() for node in edge_nodes]

            complete_graph.add_directed_edge(complete_graph.get_node(edge_nodes[0]), complete_graph.get_node(edge_nodes[1]))

        return complete_graph
    except Exception as e:
        print(f"create_new_graph::An error occurred: {str(e)}")
        # Handle the error or raise it again if needed

def get_graph_image(edge_histogram):
    try:
        complete_graph = create_new_graph(edge_histogram)

        pyd = GraphUtils.to_pydot(complete_graph)
        tmp_png = pyd.create_png(f="png")
        return tmp_png
    except Exception as e:
        print(f"get_graph_image::An error occurred: {str(e)}")

def create_graph_image(edge_histogram, save_path):
    try:
        complete_graph = create_new_graph(edge_histogram)

        pyd = GraphUtils.to_pydot(complete_graph)
        tmp_png = pyd.create_png(f="png")
        
        # Save png to path
        with open(save_path, 'wb') as f:
            f.write(tmp_png)

    except Exception as e:
        print(f"create_graph_image::An error occurred: {str(e)}")

def draw_graph(edge_histogram):
    png = get_graph_image(edge_histogram)
    fp = io.BytesIO(png)
    img = mpimg.imread(fp, format='png')
    
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.figure(figsize=(12.5, 7.5))
    plt.axis('off')
    plt.imshow(img)

    plt.show()

if __name__ == "__main__":

    # Categorize the columns of your dataframe
    categorized_data, annotation_data = readData(PARTICIPANT)

    # Apply preprocessing to the data (PCA, flattening, etc.)
    data_df = preprocess_data(categorized_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA)

    #select only data containing the features we are interested in
    data_df = data_df[[col for col in data_df.columns if any(feature in col for feature in ANALYSIS_FEATURES)]]
    column_titles = data_df.columns

    graphs = run_experiment(data_df, node_names=column_titles)
    edge_histogram = get_edge_histogram(graphs, edge_cutoff=EDGE_CUTOFF)
    print(f'Edge histogram: {edge_histogram}')
    
    draw_graph(edge_histogram)
