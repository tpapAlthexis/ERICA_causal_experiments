import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import make_scorer
from scipy import stats

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils import GraphUtils

import integrity_check
import data_acquisition as da
import globals as gl
import independence_causal as ic
from sklearn.model_selection import KFold

DATASET = gl.Dataset.RECOLA
MEASURES = [gl.AUDIO, gl.VIDEO]
FOLDS = 9

def read_data(p_to_avoid=[], apply_ica=False):   
    data, annotations = da.readDataAll_p(MEASURES, DATASET, exclude_participants=p_to_avoid)
    if not apply_ica:
        return data, annotations

    categorized_data = da.categorize_columns(data)
    #keep only the features we are interested in
    categorized_data = {key: value for key, value in categorized_data.items() if key in MEASURES}

    component_data = ic.apply_ica_to_categories(categorized_data, 0.95, 15)
    flattened_data = pd.DataFrame()
    for category, data in component_data.items():
        if isinstance(data, np.ndarray) and data.ndim == 2:  # Check if data is a 2D numpy array
            for i in range(data.shape[1]):
                new_key = f"{category}_comp_{i+1}"
                flattened_data[new_key] = data[:, i]
    return flattened_data, annotations
   
def get_comp_to_response(graph):
    if not graph:
        print("get_edge_histogram: No graphs provided.")
        return
   
    list_edges = list()
    graph_general = graph.G

    child_node_names = [gl.AROUSAL, gl.VALENCE]

    graph_edges = graph_general.get_graph_edges()
    for edge in graph_edges:
        
        point1 = edge.get_endpoint1()
        point2 = edge.get_endpoint2()

        # if it contains a bidirectional edge, skip it
        if point1 == Endpoint.ARROW and point2 == Endpoint.ARROW:
            print(f'Skipping bidirectional edge: {edge.get_node1().get_name()} -> {edge.get_node2().get_name()}')
            continue

        if edge.pointing_left(point1, point2):
            node = edge.get_node1()
        else:
            node = edge.get_node2()

        if not node:
            continue

        if node.get_name().split('_')[1] in child_node_names:
            list_edges.append(edge)

    return list_edges

def get_selected_features(edges, train_features):
    train_features_gr = None
    for edge in edges:
            if train_features_gr is None:
                train_features_gr = train_features[edge.get_node1().get_name()]
            else:
                train_features_gr = pd.concat([train_features_gr, train_features[edge.get_node1().get_name()]], axis=1)

    return train_features_gr

if __name__ == "__main__":
    p_to_avoid = integrity_check.is_ready_for_experiment(DATASET)
    if p_to_avoid:
        print(f'Experiment will be run excluding certain participants. Total participants to avoid:{len(p_to_avoid)}') 

    participants = gl.getParticipants(DATASET)
    participants = [p for p in participants if p not in p_to_avoid]

    # Initialize the SVR models
    regArousalModel = SVR()
    regValenceModel = SVR()
    causalRegArousal = SVR()
    causalRegValence = SVR()

    # Initialize the k-Fold cross-validator
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1)

    # Define the scoring metrics
    scoring = {'mse': make_scorer(mean_squared_error, greater_is_better=False), 
               'r2': make_scorer(r2_score)}

    fold_cnt = 1
    modeling_results = dict()

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(participants):
        train_participants = [participants[i] for i in train_index]
        test_participants = [participants[i] for i in test_index]

        train_features, train_targets = read_data(p_to_avoid=test_participants, apply_ica=True)
        test_features, test_targets = read_data(p_to_avoid=train_participants, apply_ica=True)

        arousal_train_targets = train_targets['median_' + gl.AROUSAL]
        valence_train_targets = train_targets['median_' + gl.VALENCE]

        arousal_test_targets = test_targets['median_' + gl.AROUSAL]
        valence_test_targets = test_targets['median_' + gl.VALENCE]

        # Train a regression model
        regArousalModel.fit(train_features, arousal_train_targets)
        regValenceModel.fit(train_features, valence_train_targets)

        # Explore causal graph
        
        # concut train features and targets to a numpy array
        train_data = pd.concat([train_features.reset_index(drop=True), train_targets.reset_index(drop=True)], axis=1)

        cg_train = pc(data=np.array(train_data),
                          alpha=0.005, #Significance level
                          indep_test=fisherz,
                          stable=True,
                          node_names=train_data.columns,
                          uc_role=2, # 0: uc_sepset, 1: maxP, 2: definateMaxP
                          uc_priority=-1) # -1: uc_role's priority, 0: overwrite, 1: orient bidirected edges, 2: ex colliders, 3: stronger colliders, 4: stronger colliders
            
        gs = cg_train.G.__str__()
        print(f'Graph string:{gs}')

        resp_edges = get_comp_to_response(cg_train)
        #drop edge from resp_edges where node1 is arousal or valence
        resp_edges = [edge for edge in resp_edges if edge.get_node1().get_name().split('_')[1] not in [gl.AROUSAL, gl.VALENCE]]
        
        arousal_edges = [edge for edge in resp_edges if edge.get_node1().get_name().split('_')[1] == gl.AROUSAL]
        valence_edges = [edge for edge in resp_edges if edge.get_node1().get_name().split('_')[1] == gl.VALENCE]

        #drop train features that are not in arousal_edges & valence_edges
        train_features_gr_arousal = get_selected_features(arousal_edges, train_features)
        train_features_gr_valence = get_selected_features(valence_edges, train_features)

        # Train the regression models
        causalRegArousal.fit(train_features_gr_arousal, arousal_train_targets)
        causalRegValence.fit(train_features_gr_valence, valence_train_targets)

        fold_cnt += 1

        #evaluate the models
        reg_arousal_predictions = regArousalModel.predict(test_features)
        reg_valence_predictions = regValenceModel.predict(test_features)

        causal_arousal_predictions = causalRegArousal.predict(test_features)
        causal_valence_predictions = causalRegValence.predict(test_features)

        #calculate kendall tau correlation
        arousal_reg_kendall, _ = stats.kendalltau(arousal_test_targets, reg_arousal_predictions)
        valence_reg_kendall, _ = stats.kendalltau(valence_test_targets, reg_valence_predictions)

        arousal_causal_kendall, _ = stats.kendalltau(arousal_test_targets, causal_arousal_predictions)
        valence_causal_kendall, _ = stats.kendalltau(valence_test_targets, causal_valence_predictions)

        print(f'Fold {fold_cnt} results:')
        print(f'- Reg Arousal Kendall:{arousal_reg_kendall}, Reg Valence Kendall:{valence_reg_kendall}')
        print(f'- Causal Arousal Kendall:{arousal_causal_kendall}, Causal Valence Kendall:{valence_causal_kendall}')
        print('-------------------------------------------------')

        modeling_results[fold_cnt] = {'reg_arousal_kendall': arousal_reg_kendall, 'reg_valence_kendall': valence_reg_kendall,
                                      'causal_arousal_kendall': arousal_causal_kendall, 'causal_valence_kendall': valence_causal_kendall}

        
    # print result summary for all folds
    print('Modeling results summary:')
    for fold, results in modeling_results.items():
        print(f'Fold {fold} results: {results}')
    
    # print mean values for all folds
    arousal_reg_kendall = np.mean([results['reg_arousal_kendall'] for results in modeling_results.values()])
    valence_reg_kendall = np.mean([results['reg_valence_kendall'] for results in modeling_results.values()])
    arousal_causal_kendall = np.mean([results['causal_arousal_kendall'] for results in modeling_results.values()])
    valence_causal_kendall = np.mean([results['causal_valence_kendall'] for results in modeling_results.values()])

    print('Mean values:')
    print(f'Reg Arousal Kendall:{arousal_reg_kendall}, Reg Valence Kendall:{valence_reg_kendall}')
    print(f'Causal Arousal Kendall:{arousal_causal_kendall}, Causal Valence Kendall:{valence_causal_kendall}')
       
        

    


    
    

    