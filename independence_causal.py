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
from sklearn.model_selection import BaseCrossValidator

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils import GraphUtils

import data_acquisition as da

LOG_SEPARATOR = '$'

DATASET_NAME = gl.Dataset.RECOLA

# Parameters
PARTICIPANT = 16 # participant number, ex participants: 16, 19, 21, 23, 25, 26, 28  
FOLDS = 5 # number of folds
COMPONENTS_THRESHOLD = 15 # number of PCA compontents. If expl. variance is lower than 95% and PCs are more, then reduct to current number
EDGE_CUTOFF = FOLDS / 2 # number of edges to be included in the histogram. If edge count is less than this number, then remove it
USE_ICA = True # use ICA instead of PCA
ANALYSIS_FEATURES = [gl.AUDIO, gl.AROUSAL, gl.VALENCE]
MEASURE_LIST = [gl.AUDIO, gl.EDA, gl.ECG, gl.VIDEO, gl.OTHER]

EXPERIMENT_SETUP = f'P{PARTICIPANT}_F{FOLDS}_C{COMPONENTS_THRESHOLD}_E{EDGE_CUTOFF}_ICA{USE_ICA}'
EXPERIMENT_ATTR = ''

expertiment_path = gl.EXPERIMENTAL_DATA_PATH + f'/independence/_P{PARTICIPANT}'
if not os.path.exists(expertiment_path):
    os.makedirs(expertiment_path)

#just to hack the cross validation
class NoSplitCV(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1  # Mimic having splits but prevent actual split

    def split(self, X, y=None, groups=None):
        indices = range(len(X))
        yield indices, indices

def preprocess_data(data_df, annotations_df, components_threshold=50, use_ica=USE_ICA, proc_logs=[''], analysis_features=None):
    categorized_data = da.categorize_columns(data_df)
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

# Function to apply PCA to each category and retain components explaining 95% variance
def apply_pca_to_categories(categorized_data, variance_threshold=0.95, components_threshold=50, proc_logs=[''], PCA_models = {}):
    pca_results = {}
    print("-------------------")
    print("PCA results")
    valid_data, invalid_categories = da.validate_categorized_data(categorized_data)
    if invalid_categories:
        print("Some categories were invalid and will be skipped:", invalid_categories)

    for category, data in valid_data.items():
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

            pca = PCA(n_components=variance_threshold, svd_solver='full')  
            pca.fit(data)
            components = pca.transform(data)

            expl_variance = np.sum(pca.explained_variance_ratio_)
            comp = components.shape[1]

            num_of_components = components_threshold if components.shape[1] > components_threshold else components.shape[1]

            if components.shape[1] > components_threshold:
                pca = PCA(n_components=components_threshold, svd_solver='full') 
                pca.fit(data)
                components = pca.transform(data)
            
            log = f"\nPCA for category: {category} - Original shape: {data.shape}, Explained variance: {expl_variance:.2f} for {comp}, Reduced to components number: {num_of_components} where expl. variance: {np.sum(pca.explained_variance_ratio_):.2f}" 
            proc_logs[0] += log
            print(log)
        except UserWarning as e:
            print(f'apply_pca_to_categories: converge failure - {str(e)}')
            proc_logs[0] += f"{LOG_SEPARATOR}PCA failed to converge for category: {category}. Category shape: {data.shape}, Number of components: {num_of_components}. Category will not contained to the analysis"
            continue

        PCA_models[category] = pca
        pca_results[category] = components

    print("-------------------" )
    return pca_results

def apply_ica_to_categories(categorized_data, variance_threshold=0.95, components_threshold=50, proc_logs=[''], ICA_models = {}):
    pca_results = {}
    ica_results = {}
    valid_data, invalid_categories = da.validate_categorized_data(categorized_data)
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
                if not category in ICA_models:
                    ica.fit(data)
                    ica_components = ica.transform(data)
                    #append ICA model to the dictionary
                    ICA_models[category] = ica
                    ica_log = f"ICA for category: {category}"
                    ica_log = f"{LOG_SEPARATOR}Components exported with: ICA{LOG_SEPARATOR}Number of iterations: {ica.n_iter_} from max iterations: 1000 ~ {"Converged" if ica.n_iter_ / 1000 < 1.00 else "Not converged!"}"
                    proc_logs[0] += f"\n{ica_log}"
                else:
                    ica_components = ICA_models[category].transform(data)
                    ica_log = f"ICA for category: {category} - Components transformed"
                if len(w) > 0 and "did not converge" in str(w[-1].message):
                    raise UserWarning("ICA failed to converge")
        except UserWarning as e:
            print(f'apply_ica_to_categories: converge failure - {str(e)}')
            proc_logs[0] += f"{LOG_SEPARATOR}ICA failed to converge for category: {category}. Category shape: {data.shape}, Number of components: {num_of_components}. Category will not contained to the analysis"
            continue
        ica_results[category] = ica_components
        #print the results if fitting has been done (not only transform)
        
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
            return None
    else:
        if groups is None:
            print("run_experiment: Cv is specified but groups are not provided.")

    fold_count = 1
    graphs = []

    if groups and len(data_df) != len(groups):
        print("Error: data_df and groups must have the same length.")
        return
    
    for train_index, test_index in cv.split(data_df, groups=groups):
        train_data = data_df.iloc[train_index].values

        # print indices of the columns that are used in the train data
        #print(f'Fold {fold_count} - Train data columns: {train_index}. Test data columns: {test_index}')
        #print(f'Fold {fold_count} - Train data shape: {train_data.shape}. Test data shape: {data_df.iloc[test_index].values.shape}')

        try:
            # Run PC algorithm on train data
            cg_train = pc(data=train_data,
                          alpha=0.005, #Significance level
                          indep_test=fisherz,
                          stable=True,
                          node_names=node_names,
                          uc_role=2, # 0: uc_sepset, 1: maxP, 2: definateMaxP
                          uc_priority=-1) # -1: uc_role's priority, 0: overwrite, 1: orient bidirected edges, 2: ex colliders, 3: stronger colliders, 4: stronger colliders
            
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
        graph_general = graph.G
        graph_edges = graph_general.get_graph_edges()

        for edge in graph_edges:
            
            point1 = edge.get_endpoint1()
            point2 = edge.get_endpoint2()

            # if it contains a bidirectional edge, skip it
            if point1 == Endpoint.ARROW and point2 == Endpoint.ARROW:
                print(f'Skipping bidirectional edge: {edge.get_node1().get_name()} <--> {edge.get_node2().get_name()}')
                continue
            if point1 != Endpoint.ARROW and point2 != Endpoint.ARROW:
                print(f'Skipping non-directed edge: {edge.get_node1().get_name()} -- {edge.get_node2().get_name()}')
                continue

            if edge.pointing_left(point1, point2):
                node = edge.get_node1()
            else:
                node = edge.get_node2()

            if not node:
                continue

            edge_str = edge.__str__()
            if edge_str in edge_histogram:
                edge_histogram[edge_str] += 1
            else:
                edge_histogram[edge_str] = 1

        fold_num += 1

    #sort histogram according to edge count
    edge_histogram = {k: v for k, v in sorted(edge_histogram.items(), key=lambda item: item[1], reverse=True)}

    #remove edges with count less than edge_cutoff
    edge_histogram = {k: v for k, v in edge_histogram.items() if v >= edge_cutoff}

    return edge_histogram

def get_nodes_from_histogram(edge_histogram):
    nodes = set()
    try:
        for edge in edge_histogram:
            edge_nodes = edge.split('-->')
            if len(edge_nodes) != 2:
                continue
            edge_nodes = [node.strip() for node in edge_nodes]
            nodes.add(edge_nodes[0])
            nodes.add(edge_nodes[1])
    except Exception as e:
        print(f"An error occurred in get_nodes_from_histogram: {str(e)}")
        return None

    return [GraphNode(node) for node in nodes]

def create_new_graph(edge_histogram):
    try:
        final_nodes = get_nodes_from_histogram(edge_histogram)

        complete_graph = GeneralGraph(nodes=final_nodes)
        for edge in edge_histogram:
            edge_nodes = edge.split('-->')
            if len(edge_nodes) != 2:
                continue
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

        # Check if file was created
        if not os.path.isfile(save_path):
            print(f"create_graph_image::Failed to create the file at: {save_path}")
            return False

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

def test_experiment():
    file_path = gl.TEST_PATH + '/data_linear_10.txt'
    if not os.path.exists(file_path):
        print(f"test_experiment: File {file_path} does not exist.")
        return
    
    data_df = pd.read_csv(file_path, sep='\t')
    data_df = data_df.dropna()

    node_names = data_df.columns

    graphs = run_experiment(data_df, cv=NoSplitCV(), node_names=node_names)
    edge_histogram = get_edge_histogram(graphs, edge_cutoff=0)
    
    draw_graph(edge_histogram)
    

if __name__ == "__main__":

    # Categorize the columns of your dataframe
    categorized_data, annotation_data = da.readData(PARTICIPANT, DATASET_NAME)

    # Apply preprocessing to the data (PCA, flattening, etc.)
    data_df = preprocess_data(categorized_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA)

    #select only data containing the features we are interested in
    data_df = data_df[[col for col in data_df.columns if any(feature in col for feature in ANALYSIS_FEATURES)]]
    column_titles = data_df.columns

    graphs = run_experiment(data_df, node_names=column_titles)
    edge_histogram = get_edge_histogram(graphs, edge_cutoff=EDGE_CUTOFF)
    print(f'Edge histogram: {edge_histogram}')
    
    draw_graph(edge_histogram)
