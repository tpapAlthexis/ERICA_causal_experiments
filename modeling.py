import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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
FOLDS = 18 #as many as RECOLA participants. Leave-one-out cross-validation

def read_data(p_to_avoid=[], apply_ica=False, ica_models={}):   
    data, annotations = da.readDataAll_p(MEASURES, DATASET, exclude_participants=p_to_avoid)
    if not apply_ica:
        selected_features = gl.Selected_audio_features + gl.Selected_video_features
        data = data[selected_features]  # Filter columns based on selected features
        return data, annotations

    categorized_data = da.categorize_columns(data)
    #keep only the features we are interested in
    categorized_data = {key: value for key, value in categorized_data.items() if key in MEASURES}

    component_data = ic.apply_ica_to_categories(categorized_data, 0.95, 15, ICA_models=ica_models)
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
        print(f'Edge: {edge.get_node1().get_name()} --> {edge.get_node2().get_name()}')
        if train_features_gr is None:
            train_features_gr = train_features[edge.get_node1().get_name()]
        else:
            train_features_gr = pd.concat([train_features_gr, train_features[edge.get_node1().get_name()]], axis=1)

    return train_features_gr

def train_model(model, features, targets):
    model.fit(features, targets)
    return model

def predict_model(model, features):
    return model.predict(features)

def calculate_kendall_tau(test_targets, predictions):
    kendall_tau, _ = stats.kendalltau(test_targets, predictions)
    return kendall_tau

def calculate_pcc(test_targets, predictions):
    pcc, _ = stats.pearsonr(test_targets, predictions)
    return pcc

def print_results(fold, *args):
    print(f'Fold {fold} results:')
    for arg in args:
        print(f'- {arg}')
    print('-------------------------------------------------')

def evaluate_baseline_model(train_f, train_t, test_f, test_t):
    if not train_f or not train_t or not test_f or not test_t:
        print("evaluate_baseline_model: Missing data.")
        return
    
    arousal_train_targets = train_targets['median_' + gl.AROUSAL]
    valence_train_targets = train_targets['median_' + gl.VALENCE]

    arousal_test_targets = test_targets['median_' + gl.AROUSAL]
    valence_test_targets = test_targets['median_' + gl.VALENCE]

    if not arousal_train_targets or not valence_train_targets or not arousal_test_targets or not valence_test_targets:
        print("evaluate_baseline_model: Missing arousal/valence targets.")
        return
    
    baselineArousalModel = LinearRegression()
    baselineValenceModel = LinearRegression()

    baselineArousalModel = train_model(baselineArousalModel, train_f, arousal_train_targets)
    baselineValenceModel = train_model(baselineValenceModel, train_f, valence_train_targets)

    baseline_arousal_predictions = predict_model(baselineArousalModel, test_f)
    baseline_valence_predictions = predict_model(baselineValenceModel, test_f)

    arousal_baseline_kendall = calculate_kendall_tau(arousal_test_targets, baseline_arousal_predictions)
    valence_baseline_kendall = calculate_kendall_tau(valence_test_targets, baseline_valence_predictions)

    arousal_baseline_pcc = calculate_pcc(arousal_test_targets, baseline_arousal_predictions)
    valence_baseline_pcc = calculate_pcc(valence_test_targets, baseline_valence_predictions)

    # return the print results
    return {'arousal_baseline_kendall': arousal_baseline_kendall, 'valence_baseline_kendall': valence_baseline_kendall,
            'arousal_baseline_pcc': arousal_baseline_pcc, 'valence_baseline_pcc': valence_baseline_pcc}

if __name__ == "__main__":
    p_to_avoid = integrity_check.is_ready_for_experiment(DATASET)
    if p_to_avoid:
        print(f'Experiment will be run excluding certain participants. Total participants to avoid:{len(p_to_avoid)}') 

    participants = gl.getParticipants(DATASET)
    participants = [p for p in participants if p not in p_to_avoid]

    FOLDS = len(participants) #as many as RECOLA participants. Leave-one-out cross-validation

    # Initialize the k-Fold cross-validator
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1)

    fold_cnt = 1
    modeling_results = dict()

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(participants):
        print("Fold number:", fold_cnt)
        # Initialize the SVR models
        regArousalModel = LinearRegression()
        regValenceModel = LinearRegression()
        causalRegArousal = LinearRegression()
        causalRegValence = LinearRegression()

        train_participants = [participants[i] for i in train_index]
        test_participants = [participants[i] for i in test_index]

        baseline_model_output = evaluate_baseline_model(train_features, train_targets, test_features, test_targets)

        ica_models = {}
        print("Reading training data...")
        train_features, train_targets = read_data(p_to_avoid=test_participants, apply_ica=True, ica_models=ica_models)
        print("Reading testing data...")
        test_features, test_targets = read_data(p_to_avoid=train_participants, apply_ica=True, ica_models=ica_models)
        print(f"Train participants len: {len(train_participants)}, Test participants len: {len(test_participants)}")

        total_training_components = train_features.shape[1]

        print("Reading data...")
        arousal_train_targets = train_targets['median_' + gl.AROUSAL]
        valence_train_targets = train_targets['median_' + gl.VALENCE]

        arousal_test_targets = test_targets['median_' + gl.AROUSAL]
        valence_test_targets = test_targets['median_' + gl.VALENCE]

        if not arousal_train_targets or not valence_train_targets or not arousal_test_targets or not valence_test_targets:
            print("Missing arousal/valence targets at fold:", fold_cnt)
            continue

        print("Training regression model...")
        regArousalModel = train_model(regArousalModel, train_features, arousal_train_targets)
        regValenceModel = train_model(regValenceModel, train_features, valence_train_targets)

        # concut train features and targets to a numpy array
        train_data = pd.concat([train_features.reset_index(drop=True), train_targets.reset_index(drop=True)], axis=1)

        print("Exploring causal graph...")
        cg_train = pc(data=np.array(train_data),
                          alpha=0.005, #Significance level
                          indep_test=fisherz,
                          stable=True,
                          node_names=train_data.columns,
                          uc_role=2, # 0: uc_sepset, 1: maxP, 2: definateMaxP
                          uc_priority=-1) # -1: uc_role's priority, 0: overwrite, 1: orient bidirected edges, 2: ex colliders, 3: stronger colliders, 4: stronger colliders
            
        gs = cg_train.G.__str__()
        print(f'Graph string:{gs}')

        print("Getting response edges...")
        resp_edges = get_comp_to_response(cg_train)
        #drop edge from resp_edges where node1 is arousal or valence
        resp_edges = [edge for edge in resp_edges if edge.get_node1().get_name().split('_')[1] not in [gl.AROUSAL, gl.VALENCE]]
        
        arousal_edges = [edge for edge in resp_edges if edge.get_node2().get_name().split('_')[1] == gl.AROUSAL]
        valence_edges = [edge for edge in resp_edges if edge.get_node2().get_name().split('_')[1] == gl.VALENCE]

        #drop train features that are not in arousal_edges & valence_edges
        train_features_gr_arousal = get_selected_features(arousal_edges, train_features)
        train_features_gr_valence = get_selected_features(valence_edges, train_features)

        total_selected_features_arousal = train_features_gr_arousal.shape[1]
        total_selected_features_valence = train_features_gr_valence.shape[1]

        if train_features_gr_arousal is None or train_features_gr_valence is None:
            print(f'No features selected for arousal or valence in fold {fold_cnt}. Skipping fold...')
            continue

        print("Training the causal regression models...")
        causalRegArousal = train_model(causalRegArousal, train_features_gr_arousal, arousal_train_targets)
        causalRegValence = train_model(causalRegValence, train_features_gr_valence, valence_train_targets)

        print("Evaluating the models for participant:", test_participants)
        reg_arousal_predictions = predict_model(regArousalModel, test_features)
        reg_valence_predictions = predict_model(regValenceModel, test_features)

        causal_test_features_arousal = get_selected_features(arousal_edges, test_features)
        causal_test_features_valence = get_selected_features(valence_edges, test_features)

        causal_arousal_predictions = predict_model(causalRegArousal, causal_test_features_arousal)
        causal_valence_predictions = predict_model(causalRegValence, causal_test_features_valence)

        print("Calculating Kendall tau correlation...")
        arousal_reg_kendall = calculate_kendall_tau(arousal_test_targets, reg_arousal_predictions)
        valence_reg_kendall = calculate_kendall_tau(valence_test_targets, reg_valence_predictions)

        arousal_causal_kendall = calculate_kendall_tau(arousal_test_targets, causal_arousal_predictions)
        valence_causal_kendall = calculate_kendall_tau(valence_test_targets, causal_valence_predictions)

        arousal_reg_pcc = calculate_pcc(arousal_test_targets, reg_arousal_predictions)
        valence_reg_pcc = calculate_pcc(valence_test_targets, reg_valence_predictions)

        arousal_causal_pcc = calculate_pcc(arousal_test_targets, causal_arousal_predictions)
        valence_causal_pcc = calculate_pcc(valence_test_targets, causal_valence_predictions)

        print_results(
            fold_cnt, 
            f'Reg Arousal Kendall: {arousal_reg_kendall}', 
            f'Reg Valence Kendall: {valence_reg_kendall}', 
            f'Causal Arousal Kendall: {arousal_causal_kendall}', 
            f'Causal Valence Kendall: {valence_causal_kendall}',
            f'Reg Arousal PCC: {arousal_reg_pcc}', 
            f'Reg Valence PCC: {valence_reg_pcc}', 
            f'Causal Arousal PCC: {arousal_causal_pcc}', 
            f'Causal Valence PCC: {valence_causal_pcc}',
            f'Training components: {total_training_components}',
            f'N features for arousal: {total_selected_features_arousal}',
            f'N features for valence: {total_selected_features_valence}'
        )
        fold_cnt += 1

        modeling_results[fold_cnt] = {'reg_arousal_kendall': arousal_reg_kendall, 'reg_valence_kendall': valence_reg_kendall,
                                      'causal_arousal_kendall': arousal_causal_kendall, 'causal_valence_kendall': valence_causal_kendall,
                                      'reg_arousal_pcc': arousal_reg_pcc, 'reg_valence_pcc': valence_reg_pcc,
                                      'causal_arousal_pcc': arousal_causal_pcc, 'causal_valence_pcc': valence_causal_pcc}

    # print result summary for all folds
    print('Modeling results summary:')
    for fold, results in modeling_results.items():
        print(f'Fold {fold} results: {results}')
    
    # print mean values for all folds
    arousal_reg_kendall = np.mean([results['reg_arousal_kendall'] for results in modeling_results.values()])
    valence_reg_kendall = np.mean([results['reg_valence_kendall'] for results in modeling_results.values()])
    arousal_causal_kendall = np.mean([results['causal_arousal_kendall'] for results in modeling_results.values()])
    valence_causal_kendall = np.mean([results['causal_valence_kendall'] for results in modeling_results.values()])
    arousal_reg_pcc = np.mean([results['reg_arousal_pcc'] for results in modeling_results.values()])
    valence_reg_pcc = np.mean([results['reg_valence_pcc'] for results in modeling_results.values()])
    arousal_causal_pcc = np.mean([results['causal_arousal_pcc'] for results in modeling_results.values()])
    valence_causal_pcc = np.mean([results['causal_valence_pcc'] for results in modeling_results.values()])

    print('Mean values:')
    print(f'Reg Arousal Kendall:{arousal_reg_kendall}, Reg Valence Kendall:{valence_reg_kendall}')
    print(f'Causal Arousal Kendall:{arousal_causal_kendall}, Causal Valence Kendall:{valence_causal_kendall}')
    print(f'Reg Arousal PCC:{arousal_reg_pcc}, Reg Valence PCC:{valence_reg_pcc}')
    print(f'Causal Arousal PCC:{arousal_causal_pcc}, Causal Valence PCC:{valence_causal_pcc}')
       
        

    


    
    

    