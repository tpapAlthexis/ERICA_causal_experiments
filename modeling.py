import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold

from datetime import datetime
import numpy as np
from sklearn.metrics import make_scorer
from scipy import stats
import os
from tabulate import tabulate
import sys

from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import Div, Spacer
from bokeh.transform import dodge

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

import pickle

def getFolderPath():
    return gl.EXPERIMENTAL_DATA_PATH + '/causal_emotion/'
EXPERIMENT_FOLDER_PATH =  getFolderPath() + 'modeling_exp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class Modeling:
    LinearRegression = 1
    MLP = 2
    SVR = 3

ModelingNames = {
    Modeling.LinearRegression: 'Linear Regression',
    Modeling.MLP: 'MLP',
    Modeling.SVR: 'SVR'
}

class ExperimentSetup:
    Default = 1
    Random_Participants = 2
    Random_P_Records = 3

CUSTOM_EXP_TITLES = {
    ExperimentSetup.Default: 'Default',
    ExperimentSetup.Random_Participants: 'Random_Participants',
    ExperimentSetup.Random_P_Records: 'Random_P_Records'
}

class DIM_REDUCTION:
    PCA = 1
    ICA = 2

DIM_REDUCTION_NAMES = {
    DIM_REDUCTION.PCA: 'PCA',
    DIM_REDUCTION.ICA: 'ICA'
}

DIM_REDUCTION_MODEL = DIM_REDUCTION.PCA
EXPERIMENT_SETUP = ExperimentSetup.Default
CUSTOM_EXP_TITLE = CUSTOM_EXP_TITLES[EXPERIMENT_SETUP]

RANDOM_PARTICIPANTS_CNT = 5
RANDOM_PARTICIPANT_PERCENTAGE = 0.25

DATASET = gl.Dataset.RECOLA
MEASURES = [gl.AUDIO, gl.VIDEO, gl.ECG, gl.EDA, gl.OTHER]
FOLDS = 18 #as many as RECOLA participants. Leave-one-out cross-validation
COMP_THRESHOLD = 5
MODELING = Modeling.LinearRegression
PARTICIPANTS = gl.getParticipants(DATASET)

# key measures enums
Graph_Metrics_str = "Graph Metrics"
class Graph_Metrics:
    Graph = 'Graph'
    TOTAL_AROUSAL_TARGETS = 'total_arousal_targets | selected'
    TOTAL_VALENCE_TARGETS = 'total_valence_targets | selected'
    TOTAL_EDGES = 'total_edges | total_selected_components'
    TOTAL_MEASURES = 'total_measures | total_components'

Model_Metrics_str = "Model Metrics"
class Model_Metrics:
    REG_AROUSAL_PCC = 'reg_arousal_pcc'
    REG_VALENCE_PCC = 'reg_valence_pcc'
    CAUSAL_AROUSAL_PCC = 'causal_arousal_pcc'
    CAUSAL_VALENCE_PCC = 'causal_valence_pcc'

Baseline_Metrics_str = "Baseline Metrics"
class Baseline_Metrics:
    AROUSAL_BASELINE_PCC = 'arousal_baseline_pcc'
    VALENCE_BASELINE_PCC = 'valence_baseline_pcc'

class ExperimentResults:
    def __init__(self, fold_results, mean_results, experiment_setup, fold_cnt, DATASET, MODELING, model_params, FOLDS, COMP_THRESHOLD, Model_Metrics_str, Model_Metrics, Baseline_Metrics_str, Baseline_Metrics, duration, graphs):
        self.mean_results = mean_results
        self.experiment_setup = experiment_setup
        self.fold_cnt = fold_cnt
        self.DATASET = DATASET
        self.MODELING = MODELING
        self.model_params = model_params
        self.fold_results = fold_results  # Stores results for each fold
        self.FOLDS = FOLDS
        self.COMP_THRESHOLD = COMP_THRESHOLD
        self.Model_Metrics_str = Model_Metrics_str
        self.Model_Metrics = Model_Metrics
        self.Baseline_Metrics_str = Baseline_Metrics_str
        self.Baseline_Metrics = Baseline_Metrics
        self.Duration = duration
        self.Graphs = graphs

        # Calculate additional metrics
        self.calculate_additional_metrics()

    def calculate_additional_metrics(self):
        self.edge_histogram = ic.get_edge_histogram(self.Graphs, FOLDS)
        self.mean_graph = ic.create_new_graph(self.edge_histogram)
        self.comp_vs_arousal_comp = self.mean_results[Graph_Metrics_str][Graph_Metrics.TOTAL_AROUSAL_TARGETS] / self.mean_results[Graph_Metrics_str][Graph_Metrics.TOTAL_MEASURES]
        self.comp_vs_valence_comp = self.mean_results[Graph_Metrics_str][Graph_Metrics.TOTAL_VALENCE_TARGETS] / self.mean_results[Graph_Metrics_str][Graph_Metrics.TOTAL_MEASURES]

        self.causal_arousal_vs_pred_arousal = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.CAUSAL_AROUSAL_PCC] / self.mean_results[self.Model_Metrics_str][self.Model_Metrics.REG_AROUSAL_PCC]
        self.causal_valence_vs_pred_valence = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.CAUSAL_VALENCE_PCC] / self.mean_results[self.Model_Metrics_str][self.Model_Metrics.REG_VALENCE_PCC]

        self.causal_arousal_vs_baseline_arousal = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.CAUSAL_AROUSAL_PCC] / self.mean_results[self.Baseline_Metrics_str][self.Baseline_Metrics.AROUSAL_BASELINE_PCC]
        self.causal_valence_vs_baseline_valence = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.CAUSAL_VALENCE_PCC] / self.mean_results[self.Baseline_Metrics_str][self.Baseline_Metrics.VALENCE_BASELINE_PCC]

        self.pred_arousal_vs_baseline_arousal = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.REG_AROUSAL_PCC] / self.mean_results[self.Baseline_Metrics_str][self.Baseline_Metrics.AROUSAL_BASELINE_PCC]
        self.pred_valence_vs_baseline_valence = self.mean_results[self.Model_Metrics_str][self.Model_Metrics.REG_VALENCE_PCC] / self.mean_results[self.Baseline_Metrics_str][self.Baseline_Metrics.VALENCE_BASELINE_PCC]

    def print_experiment_results(self):

        # Print experiment setup
        print(f'--------- Experiment setup ------------')
        print(f'Experiment title: {CUSTOM_EXP_TITLES[self.experiment_setup]}')
        print(f'Dataset: {gl.DatasetNames[self.DATASET]}')
        print(f'Modeling: {ModelingNames[self.MODELING]}')
        print(f'Model parameters: {self.model_params}')
        print(f'Folds: {self.FOLDS}')
        print(f'Components threshold: {self.COMP_THRESHOLD}')
        print(f'Measures: {MEASURES}')
        print(f'Exp. duration (m:s:ms): {self.Duration.total_seconds() // 60}m {self.Duration.total_seconds() % 60}s {self.Duration.microseconds // 1000}ms')
        print('---------------------------------------')

        # Print the results
        print('------------ Mean values --------------')
        print_results(self.mean_results)
        print('---------------------------------------')

        # Print additional metrics
        print(f'Percentage of reg performance vs baseline performance for arousal: {100.00 * self.pred_arousal_vs_baseline_arousal:.2f}')
        print(f'Percentage of components selected for arousal: {self.comp_vs_arousal_comp:.2f}')
        print(f'Percentage of causal model performance vs regression model performance for arousal: {100.00 * self.causal_arousal_vs_pred_arousal:.2f}')
        print(f'----------------------------------------------------')
        print(f'Percentage of reg performance vs baseline performance for valence: {100.00 * self.pred_valence_vs_baseline_valence:.2f}')
        print(f'Percentage of components selected for valence: {self.comp_vs_valence_comp:.2f}')
        print(f'Percentage of causal model performance vs regression model performance for valence: {100.00 * self.causal_valence_vs_pred_valence:.2f}')
        print(f'----------------------------------------------------')
        print(f'Causal arousal model VS baseline arousal model: {100.00 * self.causal_arousal_vs_baseline_arousal:.2f}')
        print(f'Causal valence model VS baseline valence model: {100.00 * self.causal_valence_vs_baseline_valence:.2f}')

def read_data(p_to_avoid=[], apply_comp_reduction=False, dim_reduction_models={}, shuffle=False, data_perc=1.0, participant_random_indices = {}):   
    data, annotations = da.readDataAll_p(MEASURES, DATASET, exclude_participants=p_to_avoid, data_percentage=data_perc, participant_random_indices=participant_random_indices)
    if not apply_comp_reduction:
        # selected_features = gl.Selected_audio_features + gl.Selected_video_features
        # data = data[selected_features]  # Filter columns based on selected features
        if shuffle:
            indices = np.random.permutation(len(data))

            data = data.iloc[indices]
            annotations = annotations.iloc[indices]

        return data, annotations

    categorized_data = da.categorize_columns(data)
    #keep only the features we are interested in
    categorized_data = {key: value for key, value in categorized_data.items() if key in MEASURES}

    if DIM_REDUCTION_MODEL == DIM_REDUCTION.PCA:
        component_data = ic.apply_pca_to_categories(categorized_data, variance_threshold=0.95, components_threshold=COMP_THRESHOLD, PCA_models=dim_reduction_models)
    elif DIM_REDUCTION_MODEL == DIM_REDUCTION.ICA:
        component_data = ic.apply_ica_to_categories(categorized_data, COMP_THRESHOLD, ICA_models=dim_reduction_models)
    else:
        print("read_data: Invalid dimensionality reduction model.")
        return

    flattened_data = pd.DataFrame()
    for category, data in component_data.items():
        if isinstance(data, np.ndarray) and data.ndim == 2:  # Check if data is a 2D numpy array
            for i in range(data.shape[1]):
                new_key = f"{category}_comp_{i+1}"
                flattened_data[new_key] = data[:, i]

    if shuffle:
        indices = np.random.permutation(len(flattened_data))

        flattened_data = flattened_data.iloc[indices]
        annotations = annotations.iloc[indices]

    return flattened_data, annotations
   
def model_init(model = MODELING):
    if model == Modeling.MLP:
        return MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1000, random_state=1, batch_size=150, shuffle=True)
    elif model == Modeling.SVR:
        return SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.1)
    else:
        return LinearRegression()

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
            train_features_gr = train_features[[edge.get_node1().get_name()]]
        else:
            train_features_gr = pd.concat([train_features_gr, train_features[[edge.get_node1().get_name()]]], axis=1)

    return train_features_gr

def train_model(model, features, targets):
    model.fit(features, targets)
    return model

def predict_model(model, features):
    return model.predict(features)

def calculate_pcc(test_targets, predictions):
    pcc, _ = stats.pearsonr(test_targets, predictions)
    return pcc

def print_results(results):
    for key, value in results.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                #if val is numeric, print with 3 decimal points
                if isinstance(sub_value, (int, float)):
                    print(f"{sub_key}: {sub_value:.3f}")
                else:
                    print(f"{sub_key}: {sub_value}")
        else:
            print(value)

def evaluate_baseline_model(train_participants, test_participants, data_perc=1.0, p_random_indices={}):
    all_participants = gl.getParticipants(DATASET)

    participants_to_avoid_train = [p for p in all_participants if p not in train_participants]
    participants_to_avoid_train.extend(test_participants)
    participants_to_avoid_test = [p for p in all_participants if p not in test_participants]
    participants_to_avoid_test.extend(train_participants)
    #remove test participant and one other participant (random) from all participants
    # all_participants = [p for p in all_participants if p not in test_participants + list(np.random.choice(all_participants, 1, replace=False))]
    train_features, train_targets = read_data(p_to_avoid=participants_to_avoid_train, apply_comp_reduction=False, data_perc=data_perc, participant_random_indices=p_random_indices)
    test_features, test_targets = read_data(p_to_avoid=participants_to_avoid_test, apply_comp_reduction=False)

    arousal_train_targets = train_targets['median_' + gl.AROUSAL]
    valence_train_targets = train_targets['median_' + gl.VALENCE]

    arousal_test_targets = test_targets['median_' + gl.AROUSAL]
    valence_test_targets = test_targets['median_' + gl.VALENCE]

    if len(arousal_train_targets) == 0 or len(valence_train_targets) == 0 or len(arousal_test_targets) == 0 or len(valence_test_targets) == 0:
        print("evaluate_baseline_model: Missing arousal/valence targets.")
        return
    
    baselineArousalModel = model_init()
    baselineValenceModel = model_init()

    baselineArousalModel = train_model(baselineArousalModel, train_features, arousal_train_targets)
    baselineValenceModel = train_model(baselineValenceModel, train_features, valence_train_targets)

    baseline_arousal_predictions = predict_model(baselineArousalModel, test_features)
    baseline_valence_predictions = predict_model(baselineValenceModel, test_features)

    arousal_baseline_pcc = calculate_pcc(arousal_test_targets, baseline_arousal_predictions)
    valence_baseline_pcc = calculate_pcc(valence_test_targets, baseline_valence_predictions)

    baseline_results = {
        Baseline_Metrics.AROUSAL_BASELINE_PCC: arousal_baseline_pcc,
        Baseline_Metrics.VALENCE_BASELINE_PCC: valence_baseline_pcc
    }

    return baseline_results

def getFoldBarPlot(exp_results, metric_str_1, metric_str_2, metric_1, metric_2, title, x_axis_label, y_axis_label, legend_title_1, legend_title_2, bar_title_1, bar_title_2):
    participants = []
    metric_1_arr = []
    metric_2_arr = []

    for fold_number, results in exp_results.fold_results.items():
        participant = results['Test Participant']['Participant']
        participants.append(f"Fold {fold_number} - P{participant}")
        metric_1_arr.append(results[metric_str_1][metric_1])
        metric_2_arr.append(results[metric_str_2][metric_2])

    source = ColumnDataSource(data={
        'participants': participants,
        bar_title_1: metric_1_arr,
        bar_title_2: metric_2_arr
    })

    bar_width = 0.4
    p = figure(x_range=participants, title=title, height=200, sizing_mode='scale_width', toolbar_location=None, tools="", x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        
    vbar1 = p.vbar(x=dodge('participants', -bar_width/2, range=p.x_range), top=bar_title_1, width=bar_width, color="blue", source=source, alpha=0.3, legend_label=legend_title_1)
    vbar2 = p.vbar(x=dodge('participants', bar_width/2, range=p.x_range), top=bar_title_2, width=bar_width, color="red", source=source, alpha=0.6, legend_label=legend_title_2)
        
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Fold # & Participant #"
        
    hover = HoverTool(renderers=[vbar1, vbar2], tooltips=[
            ("Participant", "@participants"),
            (legend_title_1, f"@{bar_title_1}"),
            (legend_title_2, f"@{bar_title_2}"),
        ], mode='vline')
    
    p.add_tools(hover)
    p.yaxis.axis_label = "Pearson Correlation Coefficient"
    p.legend.title = "Metric"
    p.legend.location = "top_left"

    return p

def create_experiment_report(exp_results, file_path):
    output_file(file_path)

    spacer = Spacer(height=30)

    exp_setup_prefs = ""
    if exp_results.experiment_setup == ExperimentSetup.Random_Participants:
        exp_setup_prefs = f"<p><b>Randomly selected participants</b>: {RANDOM_PARTICIPANTS_CNT} out of {FOLDS}</p>"
    elif exp_results.experiment_setup == ExperimentSetup.Random_P_Records:
        exp_setup_prefs = f"<p><b>Randomly selected data percentage: {RANDOM_PARTICIPANT_PERCENTAGE}</b></p>"

    exp_setup_div = Div(text=f"""
        <h1>Experiment Report for {CUSTOM_EXP_TITLES[exp_results.experiment_setup]}</h1>
        <h2>Experiment Setup</h2>
        {exp_setup_prefs}
        <p><b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Duration:</b> {(exp_results.Duration.total_seconds() // 60):.0f}m {(exp_results.Duration.total_seconds() % 60):.0f}s {exp_results.Duration.microseconds // 1000}ms</p>
        <p><b>Dataset:</b> {gl.DatasetNames[exp_results.DATASET]}</p>
        <p><b>Measures:</b> {MEASURES}</p>
        <p><b>Dim reductuion:</b> {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]}</p>
        <p><b>Modeling Technique:</b> {ModelingNames[exp_results.MODELING]}</p>
        <p><b>Model params:</b> {exp_results.model_params}</p>
        <p><b>Folds/Participants:</b> {exp_results.FOLDS}</p>
        <p><b>Component Threshold:</b> {exp_results.COMP_THRESHOLD}</p>
    """, styles={
        'style': """
            color: #333;
            font-family: Arial, sans-serif;
            """
    })

    # Prepare mean values as a string
    mean_values_str = '<h2>Final Results: Mean Values</h2>'
    categories = ['Graph Metrics', 'Model Metrics', 'Baseline Metrics']
    for category in categories:
        mean_values_str += f'<h3>{category}</h3><table style="width:100%">'
        mean_values = [(key, metric_key, metric_value) for key, value in exp_results.mean_results.items() if key == category for metric_key, metric_value in value.items()]
        for key, metric_key, metric_value in mean_values:
            if not isinstance(metric_value, (int, float)):
                mean_values_str += f'<tr><td style="text-align: right;"><b>{metric_key}:</b></td><td>{metric_value}</td></tr>'
            else:
                mean_values_str += f'<tr><td style="text-align: right;"><b>{metric_key}:</b></td><td>{metric_value:.3f}</td></tr>'
        mean_values_str += '</table>'
        
    # Create a Div widget with the mean values
    mean_values_div = Div(text=mean_values_str, styles={
        'style': """
            color: #333;
            font-family: Arial, sans-serif;
            """
    })

    # Prepare additional metrics as a string
    additional_metrics_str = '<h2>Additional Modeling Metrics</h2><table style="width:100%">'
    additional_metrics = [
        (f'Arousal | {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Reg VS Baseline', 100.00 * exp_results.pred_arousal_vs_baseline_arousal),
        ('Arousal | Causal VS Baseline', 100.00 * exp_results.causal_arousal_vs_baseline_arousal),
        ('Arousal | Components selected', exp_results.comp_vs_arousal_comp),
        ('Arousal | Causal VS Reg', 100.00 * exp_results.causal_arousal_vs_pred_arousal),
        (f'Valence | {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Reg VS Baseline', 100.00 * exp_results.pred_valence_vs_baseline_valence),
        ('Valence | Causal VS Baseline', 100.00 * exp_results.causal_valence_vs_baseline_valence),
        ('Valence | Components selected', exp_results.comp_vs_valence_comp),
        ('Valence | Causal VS Reg', 100.00 * exp_results.causal_valence_vs_pred_valence)
    ]
    for metric_name, metric_value in additional_metrics:
        if not isinstance(metric_value, (int, float)):
            additional_metrics_str += f'<tr><td style="text-align: right;"><b>{metric_name}:</b></td><td>{metric_value}</td></tr>'
        else:
            additional_metrics_str += f'<tr><td style="text-align: right;"><b>{metric_name}:</b></td><td>{metric_value:.2f}%</td></tr>'
    additional_metrics_str += '</table>'

    # Create a Div widget with the additional metrics
    additional_metrics_div = Div(text=additional_metrics_str, styles={
        'style': """
            color: #333;
            font-family: Arial, sans-serif;
            """
    })

    p_c_c = getFoldBarPlot(exp_results, 'Model Metrics', 'Model Metrics', Model_Metrics.CAUSAL_AROUSAL_PCC, Model_Metrics.CAUSAL_VALENCE_PCC, "Arousal VS Valence Causal Modeling", "Fold Participant", "PCC Value", "Causal Arousal Model", "Causal Valence Model", "causal_arousal_pcc", "causal_valence_pcc")
    p_baseline_vs_comp_modeling_arousal = getFoldBarPlot(exp_results, 'Model Metrics', 'Baseline Metrics', Model_Metrics.REG_AROUSAL_PCC, Baseline_Metrics.AROUSAL_BASELINE_PCC, f"Baseline VS {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Arousal", "Fold Participant", "PCC Value", f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Model", "Baseline Model", "reg_arousal_pcc", "arousal_baseline_pcc")
    p_baseline_vs_comp_modeling_valence = getFoldBarPlot(exp_results, 'Model Metrics', 'Baseline Metrics', Model_Metrics.REG_VALENCE_PCC, Baseline_Metrics.VALENCE_BASELINE_PCC, f"Baseline VS {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Valence", "Fold Participant", "PCC Value", f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Model", "Baseline Model", "reg_valence_pcc", "valence_baseline_pcc")
    p_comp_modeling_vs_causal_arousal = getFoldBarPlot(exp_results, 'Model Metrics', 'Model Metrics', Model_Metrics.REG_AROUSAL_PCC, Model_Metrics.CAUSAL_AROUSAL_PCC, f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} VS Causal {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Arousal", "Fold Participant", "PCC Value", f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Model", "Causal Model", "reg_arousal_pcc", "causal_arousal_pcc")
    p_comp_modeling_vs_causal_valence = getFoldBarPlot(exp_results, 'Model Metrics', 'Model Metrics', Model_Metrics.REG_VALENCE_PCC, Model_Metrics.CAUSAL_VALENCE_PCC, f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} VS Causal {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Valence", "Fold Participant", "PCC Value", f"{DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Model", "Causal Model", "reg_valence_pcc", "causal_valence_pcc")
    p_baseline_vs_causal_arousal = getFoldBarPlot(exp_results, 'Model Metrics', 'Baseline Metrics', Model_Metrics.CAUSAL_AROUSAL_PCC, Baseline_Metrics.AROUSAL_BASELINE_PCC, f"Baseline VS Causal {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Arousal", "Fold Participant", "PCC Value", "Causal Model", "Baseline Model", "causal_arousal_pcc", "arousal_baseline_pcc")
    p_baseline_vs_causal_valence = getFoldBarPlot(exp_results, 'Model Metrics', 'Baseline Metrics', Model_Metrics.CAUSAL_VALENCE_PCC, Baseline_Metrics.VALENCE_BASELINE_PCC, f"Baseline VS Causal {DIM_REDUCTION_NAMES[DIM_REDUCTION_MODEL]} Modeling ~ Valence", "Fold Participant", "PCC Value", "Causal Model", "Baseline Model", "causal_valence_pcc", "valence_baseline_pcc")

    layout = column(exp_setup_div, mean_values_div, additional_metrics_div, spacer, p_c_c, p_baseline_vs_comp_modeling_arousal, p_baseline_vs_comp_modeling_valence, p_baseline_vs_causal_arousal, p_baseline_vs_causal_valence, p_comp_modeling_vs_causal_arousal, p_comp_modeling_vs_causal_valence, sizing_mode='stretch_width')
    save(layout)

def create_experiment_folder_path(exp):
    if not os.path.exists(EXPERIMENT_FOLDER_PATH):
        os.makedirs(EXPERIMENT_FOLDER_PATH)

    with open(EXPERIMENT_FOLDER_PATH + '/experiment_results.pkl', 'wb') as f:
        pickle.dump(experiment, f)

    return EXPERIMENT_FOLDER_PATH

def exportParticipants():
    participants = gl.getParticipants(DATASET)
    p_to_avoid = integrity_check.is_ready_for_experiment(DATASET)
    participants = [p for p in participants if p not in p_to_avoid]

    csv_data = []

    for participant in participants:
        part_avoid = [p for p in participants if p != participant]
        data, annotations = read_data(p_to_avoid=part_avoid, apply_comp_reduction=False)

        #populate participant column
        part_df = pd.DataFrame([participant] * len(data), columns=['Participant'])
        #append data and annotations for each participant. Assign a new column for participant number
        data_frame = pd.concat([part_df, data, annotations], axis=1)

        csv_data.append(data_frame)
    
    #concatenate all dataframes
    final_data = pd.concat(csv_data)
    #set titles for the columns
    final_data.columns = ['Participant'] + list(final_data.columns[1:])

    final_data.to_csv(gl.EXPERIMENTAL_DATA_PATH + '/all_participants_data.csv', index=False)
        

def runExperiment(exp_setup=ExperimentSetup.Default):
    start_time = datetime.now()

    p_to_avoid = integrity_check.is_ready_for_experiment(DATASET)
    if p_to_avoid:
        print(f'Experiment will be run excluding certain participants. Total participants to avoid:{len(p_to_avoid)}') 

    participants = gl.getParticipants(DATASET)
    participants = [p for p in participants if p not in p_to_avoid]

    FOLDS = len(participants) #as many as RECOLA participants. Leave-one-out cross-validation

    # Initialize the k-Fold cross-validator
    kf = KFold(n_splits=FOLDS, shuffle=True)

    fold_cnt = 0
    modeling_results = dict()

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(participants):
        fold_cnt += 1
        print("Fold number:", fold_cnt)
        # Initialize the SVR models
        regArousalModel = model_init()
        regValenceModel = model_init()
        causalRegArousal = model_init()
        causalRegValence = model_init()

        train_participants = [participants[i] for i in train_index]
        test_participants = [participants[i] for i in test_index]

        print(f'Train participants: {train_participants}, Test participants: {test_participants}')

        dim_reduction_models = {}
        participant_data_perc = 1.0

        if (exp_setup == ExperimentSetup.Random_Participants):
            train_participants = np.random.choice(train_participants, RANDOM_PARTICIPANTS_CNT, replace=False)
            print(f'Randomly selected participants: {train_participants}')
        elif (exp_setup == ExperimentSetup.Random_P_Records):
            participant_data_perc = RANDOM_PARTICIPANT_PERCENTAGE
            print(f'Randomly selected data percentage: {participant_data_perc}')

        all_participants = gl.getParticipants(DATASET)
        participants_to_avoid_train = [p for p in all_participants if p not in train_participants]
        participants_to_avoid_train.extend(test_participants)
        participants_to_avoid_test = [p for p in all_participants if p not in test_participants]
        participants_to_avoid_test.extend(train_participants)

        p_random_indices = {}

        print("Reading training data...")
        train_features, train_targets = read_data(p_to_avoid=participants_to_avoid_train, apply_comp_reduction=True, dim_reduction_models=dim_reduction_models, data_perc=participant_data_perc, participant_random_indices=p_random_indices)

        print("Reading testing data...")
        test_features, test_targets = read_data(p_to_avoid=participants_to_avoid_test, apply_comp_reduction=True, dim_reduction_models=dim_reduction_models)
        print(f"Train participants len: {len(train_participants)}, Test participants len: {len(test_participants)}")

        # if test features cols are less than train features cols, drop the extra cols
        if len(test_features.columns) > len(train_features.columns):
            print(f"Dropping extra columns from test features: {len(test_features.columns) - len(train_features.columns)}")
            test_features = test_features.drop(columns=[col for col in test_features.columns if col not in train_features.columns])
        if len(train_features.columns) > len(test_features.columns):
            print(f"Dropping extra columns from train features: {len(train_features.columns) - len(test_features.columns)}")
            train_features = train_features.drop(columns=[col for col in train_features.columns if col not in test_features.columns])

        print("Evaluating baseline model...")
        baseline_results = evaluate_baseline_model(train_participants, test_participants, data_perc=participant_data_perc, p_random_indices=p_random_indices)
        print(f'Baseline model finished.')

        print("Reading data...")
        arousal_train_targets = train_targets['median_' + gl.AROUSAL]
        valence_train_targets = train_targets['median_' + gl.VALENCE]

        arousal_test_targets = test_targets['median_' + gl.AROUSAL]
        valence_test_targets = test_targets['median_' + gl.VALENCE]

        if len(arousal_train_targets) == 0 or len(valence_train_targets) == 0:
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

        if train_features_gr_arousal is None or train_features_gr_valence is None:
            print(f'No features selected for arousal or valence in fold {fold_cnt}. Skipping fold...')
            continue

        print("Training the causal regression models...")
        causalRegArousal = train_model(causalRegArousal, train_features_gr_arousal, arousal_train_targets)
        causalRegValence = train_model(causalRegValence, train_features_gr_valence, valence_train_targets)

        if not 'video_comp_4' in test_features.columns:
            print(f'video_comp_4 not in train_features.columns. Skipping fold {fold_cnt}...')

        print("Evaluating the models for participant:", test_participants)
        reg_arousal_predictions = predict_model(regArousalModel, test_features)
        reg_valence_predictions = predict_model(regValenceModel, test_features)

        causal_test_features_arousal = get_selected_features(arousal_edges, test_features)
        causal_test_features_valence = get_selected_features(valence_edges, test_features)

        causal_arousal_predictions = predict_model(causalRegArousal, causal_test_features_arousal)
        causal_valence_predictions = predict_model(causalRegValence, causal_test_features_valence)

        print("Calculating PCC...")
        arousal_reg_pcc = calculate_pcc(arousal_test_targets, reg_arousal_predictions)
        valence_reg_pcc = calculate_pcc(valence_test_targets, reg_valence_predictions)

        arousal_causal_pcc = calculate_pcc(arousal_test_targets, causal_arousal_predictions)
        valence_causal_pcc = calculate_pcc(valence_test_targets, causal_valence_predictions)

        print("Finished evaluating the models...")

        graph_results = {
            Graph_Metrics.Graph: cg_train,
            Graph_Metrics.TOTAL_AROUSAL_TARGETS: len(arousal_edges),
            Graph_Metrics.TOTAL_VALENCE_TARGETS: len(valence_edges),
            Graph_Metrics.TOTAL_EDGES: len(resp_edges),
            Graph_Metrics.TOTAL_MEASURES: len(train_features.columns)
        }

        prediction_results = {
            Model_Metrics.REG_AROUSAL_PCC: arousal_reg_pcc,
            Model_Metrics.REG_VALENCE_PCC: valence_reg_pcc,
            Model_Metrics.CAUSAL_AROUSAL_PCC: arousal_causal_pcc,
            Model_Metrics.CAUSAL_VALENCE_PCC: valence_causal_pcc
        }

        results_summary = {
            'Test Participant': {'Participant': test_participants},
            'Graph Metrics': graph_results,
            'Model Metrics': prediction_results,
            'Baseline Metrics': baseline_results
        }

        print(f'------------ Results@fold {fold_cnt} --------------')
        print_results(results_summary)
        print('----------------------------------------------------')

        modeling_results[fold_cnt] = results_summary

    end_time = datetime.now()

    mean_results = {}
    graphs = []
    for fold, value in modeling_results.items():
        if not isinstance(value, dict):
            print(f'Value {value} is not hashable. Skipping...')
            continue
        for sub_key, sub_value in value.items():
            if sub_key not in mean_results:
                mean_results[sub_key] = {}
            for metric_key, metric_value in sub_value.items():
                if metric_key is Graph_Metrics.Graph:
                    graphs.append(metric_value)
                    continue
                if not isinstance(metric_value, (int, float)):
                    mean_results.pop(sub_key, None)
                    continue
                if metric_key not in mean_results[sub_key]:
                    mean_results[sub_key][metric_key] = 0
                mean_results[sub_key][metric_key] += metric_value

    print(f'Folds: {fold_cnt}')
    for key, value in mean_results.items():
        for metric_key, metric_value in value.items():
            print(f'{key} - {metric_key}: {metric_value} -> assign: {metric_value / fold_cnt}')
            mean_results[key][metric_key] = metric_value / fold_cnt

    model = model_init()
    model_params = model.get_params()

    experiment = ExperimentResults(
    fold_results=modeling_results,  # This is the dictionary with all fold results
    mean_results=mean_results,      # Assuming you have a dictionary with mean values
    experiment_setup=exp_setup,
    fold_cnt=fold_cnt,
    DATASET=DATASET,
    MODELING=MODELING,
    model_params=model_params,
    FOLDS=FOLDS,
    COMP_THRESHOLD=COMP_THRESHOLD,
    Model_Metrics_str=Model_Metrics_str,
    Model_Metrics=Model_Metrics,
    Baseline_Metrics_str=Baseline_Metrics_str,
    Baseline_Metrics=Baseline_Metrics,
    duration=end_time - start_time,
    graphs=graphs
    )
    experiment.print_experiment_results()

    with open('experiment_results.pkl', 'wb') as f:
        pickle.dump(experiment, f)

    return experiment

def saveGraphImages(experiment, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for fold, value in experiment.fold_results.items():
        if not isinstance(value, dict):
            print(f'saveGraphImages: Value {value} is not hashable. Skipping...')
            continue
        for sub_key, sub_value in value.items():
            for metric_key, metric_value in sub_value.items():
                if metric_key == Graph_Metrics.Graph:
                    graph = metric_value
                    graph_name = f'graph_fold_{fold}_p_{str(value['Test Participant']['Participant'])}.jpg'
                    pyd = GraphUtils.to_pydot(graph.G)
                    tmp_png = pyd.create_png(f="png")

                    savePath = path + graph_name
                    
                    # Save png to path
                    with open(savePath, 'wb') as f:
                        f.write(tmp_png)

                    if not os.path.isfile(savePath):
                        print(f"create_graph_image::Failed to create the file at: {savePath}")
                        return False

if __name__ == "__main__":

    EXPERIMENT_FOLDER_PATH = getFolderPath() + 'modeling_exp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_{CUSTOM_EXP_TITLE}'

    experiment = None
    experiment = runExperiment(EXPERIMENT_SETUP)

    if not experiment:
        with open('experiment_results.pkl', 'rb') as f:
            experiment = pickle.load(f)

    create_experiment_folder_path(exp=experiment)
    saveGraphImages(experiment, EXPERIMENT_FOLDER_PATH + '/graphs/')
    create_experiment_report(experiment, file_path=f'{EXPERIMENT_FOLDER_PATH}/modeling.html')