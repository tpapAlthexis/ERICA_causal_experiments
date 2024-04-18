import independence_causal as ic
import globals as gl
from enum import Enum
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import base64
import numpy as np
from sklearn.model_selection import GroupKFold

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import layout, column
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.io import curdoc
from bokeh.models.widgets import Div

import integrity_check

PARTICIPANTS = gl.getParticipants()
COMPONENTS_THRESHOLD = ic.COMPONENTS_THRESHOLD
USE_ICA = True
FOLDS = 9
EDGE_CUTOFF = int(FOLDS / 2)
EXPERIMENT_FOLDER_PATH = gl.EXPERIMENTAL_DATA_PATH + '/causal_emotion/' + 'exp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RUN_FOR_ALL_PARTICIPANTS = True
EXPERIMENT_MEASURES = [gl.AUDIO]#[gl.AUDIO, gl.EDA, gl.ECG, gl.VIDEO, gl.OTHER]
ALL_P_GRAPH_POSTFIX = '_all_p_graph'
DATASET_NAME = gl.DatasetName.SEWA

class ExperimentEnum(Enum):
    Setup = 0
    Preproc_logs = 1
    Graphs = 2

class ExperimentSetup(Enum):
    Participant = 0
    Folds = 1
    Use_ICA = 2
    Components_Threshold = 3
    Edge_Cutoff = 4
    Analysis_Features = 5

def create_comp_plot_image(graph_plots, feat, path):
    images = []
    for participant in graph_plots:
        img_data = graph_plots[participant]
        img = Image.open(BytesIO(img_data))  
        font = ImageFont.truetype('arial.ttf', 45) 
        images.append((participant, img))

    # Create a new image with the height equal to the sum of all image heights and line images
    total_height = sum(img.height for participant, img in images) + 100 * (len(images) + 1)
    max_width = max(img.width for participant, img in images)
    composed_image = Image.new('RGBA', (max_width, total_height), 'white') 

    # Add measure title to the top of the composed image
    title_image = Image.new('RGBA', (max_width, 100), 'white') 
    title_draw = ImageDraw.Draw(title_image)
    title_font = ImageFont.truetype('arialbd.ttf', 60)
    title_text = f'Measure: {feat}'
    title_width, title_height = [200, 100]
    title_draw.text(((max_width - title_width) / 2, (100 - title_height) / 2), title_text, fill='black', font=title_font)
    composed_image.paste(title_image, (0, 0))

    # Append images vertically
    y_offset = title_image.height
    for participant, img in images:
        # Add a horizontal line between images with participant title
        line_image = Image.new('RGBA', (max_width, 100), 'white')  # Create a larger white image
        line_draw = ImageDraw.Draw(line_image)
        line_draw.line((0, 0, max_width, 0), fill='black')  # Draw a black line at the top

        # Add participant title to the line image
        line_draw.text((10, 50), f'Participant: {participant}', fill='black', font=font)

        composed_image.paste(line_image, (0, y_offset))
        y_offset += line_image.height

        # Paste the image of the participant's plot
        composed_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the composed image
    composed_image.save(os.path.join(path, f'{feat}_composed.png'))

def create_experiment_folder_path():
    if not os.path.exists(EXPERIMENT_FOLDER_PATH):
        os.makedirs(EXPERIMENT_FOLDER_PATH)

    return EXPERIMENT_FOLDER_PATH

def run_causal_emotion_experiment(participant, analysis_features):
    measure_data, annotation_data = ic.readData(participant)

    exp_dict = {}
    exp_setup = {ExperimentSetup.Participant.name: participant, ExperimentSetup.Folds.name: FOLDS, ExperimentSetup.Use_ICA.name: USE_ICA, ExperimentSetup.Components_Threshold.name: COMPONENTS_THRESHOLD, ExperimentSetup.Edge_Cutoff.name: EDGE_CUTOFF, ExperimentSetup.Analysis_Features.name: analysis_features}
    exp_dict[ExperimentEnum.Setup.name] = exp_setup
    exp_dict[ExperimentEnum.Preproc_logs.name] = ['']
    # Apply preprocessing to the data (categorization, ICA, flattening, etc.)
    data_df = ic.preprocess_data(measure_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA, exp_dict[ExperimentEnum.Preproc_logs.name], analysis_features)

    graphs = ic.run_experiment(data_df, folds=FOLDS, node_names=data_df.columns)
    exp_dict[ExperimentEnum.Graphs.name] = graphs

    return exp_dict

def run_experiment_for_all_p(analysis_features):
    measure_data, annotation_data = ic.readDataAll_p(analysis_features, DATASET_NAME)

    if measure_data is None or annotation_data is None:
        print('No data found for all participants. Aborting experiment.')
        return None

    exp_dict = {}
    exp_setup = {ExperimentSetup.Participant.name: "All participants", ExperimentSetup.Folds.name: FOLDS, ExperimentSetup.Use_ICA.name: USE_ICA, ExperimentSetup.Components_Threshold.name: COMPONENTS_THRESHOLD, ExperimentSetup.Edge_Cutoff.name: EDGE_CUTOFF, ExperimentSetup.Analysis_Features.name: analysis_features}
    exp_dict[ExperimentEnum.Setup.name] = exp_setup
    exp_dict[ExperimentEnum.Preproc_logs.name] = ['']

    # groupKfold where each group is participant's data length - assuming that annotations match the measure data
    groups_sizes = [len((pd.read_csv(gl.getAnnotationsPath(participant, DATASET_NAME)))) for participant in gl.getParticipants(DATASET_NAME)]
    groups = [i for i in range(len(groups_sizes)) for _ in range(groups_sizes[i])]

    cv_g = GroupKFold(n_splits=FOLDS)

    # Apply preprocessing to the data (categorization, ICA, flattening, etc.)
    data_df = ic.preprocess_data(measure_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA, exp_dict[ExperimentEnum.Preproc_logs.name], analysis_features)
    exp_folds = 9
    for feat in analysis_features:
        exp_dict[feat] = {}
        #select data for the feature + arousal and valence
        exp_data_df = ic.get_category_features(data_df, feat, True)
        graphs = ic.run_experiment(exp_data_df, folds=exp_folds, node_names=exp_data_df.columns, cv=cv_g, groups=groups)
        if graphs is None:
            print(f'No graphs detected for feature {feat}')
            continue
        exp_dict[feat][ExperimentEnum.Graphs.name] = graphs
        
        edge_histogram = ic.get_edge_histogram(exp_dict[feat][ExperimentEnum.Graphs.name], EDGE_CUTOFF)
        ic.create_graph_image(edge_histogram, os.path.join(EXPERIMENT_FOLDER_PATH, f'{feat}{ALL_P_GRAPH_POSTFIX}.png'))

    return exp_dict

def run_experiment(features = EXPERIMENT_MEASURES):
    const_features = [ic.AROUSAL, ic.VALENCE] 

    experiment_res_dict = {}
    path = create_experiment_folder_path()

    for feat in features:
        graph_plots = {}
        experiment_res_dict[feat] = {}
        for participant in PARTICIPANTS:
            analysis_features = const_features + [feat]
            experiment_res_dict[feat][participant] = run_causal_emotion_experiment(participant, analysis_features)

            current_experiment = experiment_res_dict[feat][participant]

            edge_histogram = ic.get_edge_histogram(current_experiment[ExperimentEnum.Graphs.name], EDGE_CUTOFF)
            graph_plots[participant] = ic.get_graph_image(edge_histogram)

        create_comp_plot_image(graph_plots, feat, path)

    return experiment_res_dict

def create_save_histograms(res, path):
    for feat in res:
        edge_histogram = {}
        participant_values = []
        for participant in res[feat]:
            current_experiment = res[feat][participant]
            #get all edge data wihout cutoffs
            tmp_edge_histogram = ic.get_edge_histogram(current_experiment[ExperimentEnum.Graphs.name], 0)
            #append the edge_histograms
            for edge in tmp_edge_histogram:
                if edge in edge_histogram:
                    edge_histogram[edge] += tmp_edge_histogram[edge]
                else:
                    edge_histogram[edge] = tmp_edge_histogram[edge]

            if len(tmp_edge_histogram) > 0:
                participant_values.append(sum(tmp_edge_histogram.values()) / len(tmp_edge_histogram))
            else:
                print(f'Participant {participant} has no detected directional edges for feature {feat}')
                participant_values.append(0)

        max_edge_count = len(res[feat]) * FOLDS
        scaled_edge_histogram = {k: v / max_edge_count for k, v in edge_histogram.items()}

        # Calculate the mean and standard deviation frequencies for the feature
        mean = np.mean(participant_values) / max_edge_count
        std_dev = np.std(participant_values) / max_edge_count
        
        scaled_edge_histogram = {k: v for k, v in sorted(scaled_edge_histogram.items(), key=lambda item: item[1], reverse=True)}

        #create a histogram image for the feature
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scaled_edge_histogram.keys(), scaled_edge_histogram.values(), color='b')
        plt.xlabel('Edges')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for {feat}')
        plt.grid(True)

        # Print labels vertically inside each corresponding bin
        for bar, label in zip(bars, scaled_edge_histogram.keys()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, label, ha='center', va='bottom', rotation=90, fontsize=8, color='red', weight='bold')

        # Print mean and standard deviation
        plt.text(0.02, 0.95, f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        plt.xticks([])  # Make x-axis labels invisible

        plt.savefig(os.path.join(path, f'{feat}_histogram.png'))  # Save the figure before showing it
        plt.show()
        plt.close()

        edge_histogram = {k: v for k, v in edge_histogram.items() if v >= max_edge_count / 2}
        ic.draw_graph(edge_histogram)

def create_save_histograms_all_p(res, path):
    for feat in res:
        if not feat in [ic.AUDIO, ic.ECG, ic.VIDEO, ic.EDA, ic.OTHER]:
            continue 

        edge_histogram = {}
       
        #get all edge data wihout cutoffs
        tmp_edge_histogram = ic.get_edge_histogram(res[feat][ExperimentEnum.Graphs.name], 0)
        #append the edge_histograms
        for edge in tmp_edge_histogram:
            if edge in edge_histogram:
                edge_histogram[edge] += tmp_edge_histogram[edge]
            else:
                edge_histogram[edge] = tmp_edge_histogram[edge]

        max_edge_count = len(res[feat]) * FOLDS
        scaled_edge_histogram = {k: v / max_edge_count for k, v in edge_histogram.items()}
        
        scaled_edge_histogram = {k: v for k, v in sorted(scaled_edge_histogram.items(), key=lambda item: item[1], reverse=True)}

        #create a histogram image for the feature
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scaled_edge_histogram.keys(), scaled_edge_histogram.values(), color='b')
        plt.xlabel('Edges')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for {feat}')
        plt.grid(True)

        # Print labels vertically inside each corresponding bin
        for bar, label in zip(bars, scaled_edge_histogram.keys()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, label, ha='center', va='bottom', rotation=90, fontsize=8, color='red', weight='bold')

        plt.xticks([])  # Make x-axis labels invisible

        plt.savefig(os.path.join(path, f'{feat}_histogram.png'), dpi=300)  # Save the figure before showing it with higher resolution
        plt.show()
        plt.close()

        edge_histogram = {k: v for k, v in edge_histogram.items() if v >= max_edge_count / 2}
        ic.draw_graph(edge_histogram)

def create_image_div(image_path, page_width=900):
    # Open the image with PIL
    img = Image.open(image_path)

    # Get the width and height of the image
    original_width, original_height = img.size

    # Calculate the new width that maintains the aspect ratio with a maximum height of 400
    new_height = 300
    new_width = int(original_width * new_height / original_height)

    # If the new width is greater than the page width, recalculate the new width and height
    if new_width > page_width:
        new_width = page_width
        new_height = int(original_height * new_width / original_width)

    # Resize the image
    img = img.resize((new_width, new_height))

    # Convert the resized image to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Create a Div widget with an img tag that uses the base64 string as the source
    image_div = Div(text=f'<img src="data:image/png;base64,{image_b64}" width="{new_width}" height="{new_height}"/>')

    return image_div

def create_experiment_report(exp_dict, path):
    # Set the output file
    output_file(os.path.join(path, "experiment_report.html"))

    # Create the document title and setup information
    experiment_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    doc_title = Div(text=f"<h1>Experiment Datetime: {experiment_datetime}</h1>")
    
    setup_text = "<h2>Experiment Setup:</h2>"
    for key, value in exp_dict['Setup'].items():
        setup_text += f"<b>{key}:</b> {value}<br>"
    
    experiment_setup = Div(text=setup_text)
    
    preproc_logs_text = "<h2>Preprocessing Logs:</h2>"
    for log in exp_dict['Preproc_logs']:
        records = log.split(ic.LOG_SEPARATOR)
        for record in records:
            parts = record.split(':')
            if len(parts) == 2:
                preproc_logs_text += f"<b>{parts[0]}:</b> {parts[1]}<br>"
            else:
                preproc_logs_text += f"{record}<br>"
    preproc_logs = Div(text=preproc_logs_text)

    # List to hold all sections for layout
    layout_sections = [doc_title, experiment_setup, preproc_logs]

    # Process each measure
    for feat in exp_dict:
        if feat == 'Setup' or feat == 'Preproc_logs':
            continue  # Skip non-measure data

        # Process each measure data
        edge_histogram = ic.get_edge_histogram(res[feat][ExperimentEnum.Graphs.name], 0)

        # a) Histogram data in text
        histogram_text = f"<h2>Histogram Data for {feat}:</h2>"
        for edge, count in edge_histogram.items():
            normalized_value = count / FOLDS
            histogram_text += f"Edge: {edge}, Count: {count}, Fold Normalized: {normalized_value:.2f}<br>"
        
        histogram_div = Div(text=histogram_text)
        layout_sections.append(histogram_div)

        # b) Interactive histogram with Bokeh
        edge_histogram = ic.get_edge_histogram(res[feat][ExperimentEnum.Graphs.name], EDGE_CUTOFF)
        edges = list(edge_histogram.keys())
        counts = list(edge_histogram.values())
        source = ColumnDataSource(data=dict(edges=edges, counts=counts, half_counts=[count / 2 for count in counts]))

        p = figure(x_range=edges, height=200, sizing_mode='scale_width', title=f"Histogram chart (applied cutoff):",
                toolbar_location=None, tools="", y_axis_label='Edge count')
        bars = p.vbar(x='edges', top='counts', width=0.9, source=source)

        p.y_range.start = 0
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_text_font_size = '0pt'  # Hide x-axis labels
        p.add_tools(HoverTool(tooltips=[("Edge", "@edges"), ("Count", "@counts")]))

        # Create a LabelSet and add it to the plot
        labels = LabelSet(x='edges', y='half_counts', text='edges', level='glyph',
                        x_offset=-13.5, y_offset=0, source=source, text_font_size='8pt', text_color='white', text_align='center', angle=3.14/2)
        p.add_layout(labels)

        layout_sections.append(p)

        # c) The graph image
        graph_image_path = os.path.join(path, f'{feat}{ALL_P_GRAPH_POSTFIX}.png')  # Adjust path as necessary
        image_div = create_image_div(graph_image_path, page_width=1200)

        # Append the Div widget to the layout sections
        layout_sections.append(Div(text=f"<h2>Graph Image for {feat}:</h2>"))
        layout_sections.append(image_div)

    # Finalize layout and save
    l = column(layout_sections, sizing_mode='scale_width')
    curdoc().add_root(l)
    save(l)

if __name__ == "__main__":

    if integrity_check.is_ready_for_experiment(DATASET_NAME) == False:
        print('Experiment cannot be run. Data is not ready.')
        exit()

    path = create_experiment_folder_path()

    if RUN_FOR_ALL_PARTICIPANTS:
        res = run_experiment_for_all_p(EXPERIMENT_MEASURES)
        # create_save_histograms_all_p(res, path)
        create_experiment_report(res, path)
        print(f'Experimental setup: {res[ExperimentEnum.Setup.name]}')
        print(f'Experiment preprocessing logs: {res[ExperimentEnum.Preproc_logs.name]}')
        print('Experiment for all participants finished')
    else:
        res = run_experiment(EXPERIMENT_MEASURES)
        create_save_histograms(res, path)

    

    