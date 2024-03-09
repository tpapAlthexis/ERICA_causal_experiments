import independence_causal as ic
import globals as gl
from enum import Enum
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
import sys
import numpy as np

PARTICIPANTS = [64,65]#gl.getParticipants()
COMPONENTS_THRESHOLD = ic.COMPONENTS_THRESHOLD
USE_ICA = True
FOLDS = ic.FOLDS
EDGE_CUTOFF = FOLDS / 2
EXPERIMENT_FOLDER_PATH = gl.EXPERIMENTAL_DATA_PATH + '/causal_emotion/' + 'exp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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
        img = Image.open(BytesIO(img_data))  # Convert bytes to image
        font = ImageFont.truetype('arial.ttf', 45)  # You may need to adjust the font path and size
        images.append((participant, img))

    # Create a new image with the height equal to the sum of all image heights and line images
    total_height = sum(img.height for participant, img in images) + 100 * (len(images) + 1)
    max_width = max(img.width for participant, img in images)
    composed_image = Image.new('RGBA', (max_width, total_height), 'white')  # Use white background

    # Add measure title to the top of the composed image
    title_image = Image.new('RGBA', (max_width, 100), 'white')  # Create a larger white image
    title_draw = ImageDraw.Draw(title_image)
    title_font = ImageFont.truetype('arialbd.ttf', 60)  # You may need to adjust the font path and size
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
    categorized_data, annotation_data = ic.readData(participant)

    exp_dict = {}
    exp_setup = {ExperimentSetup.Participant.name: participant, ExperimentSetup.Folds.name: FOLDS, ExperimentSetup.Use_ICA.name: USE_ICA, ExperimentSetup.Components_Threshold.name: COMPONENTS_THRESHOLD, ExperimentSetup.Edge_Cutoff.name: EDGE_CUTOFF, ExperimentSetup.Analysis_Features.name: analysis_features}
    exp_dict[ExperimentEnum.Setup.name] = exp_setup
    exp_dict[ExperimentEnum.Preproc_logs.name] = ['']
    # Apply preprocessing to the data (PCA, flattening, etc.)
    data_df = ic.preprocess_data(categorized_data, annotation_data, COMPONENTS_THRESHOLD, USE_ICA, exp_dict[ExperimentEnum.Preproc_logs.name])

    #select only data containing the features we are interested in
    data_df = data_df[[col for col in data_df.columns if any(feature in col for feature in analysis_features)]]

    graphs = ic.run_experiment(data_df, folds=FOLDS, node_names=data_df.columns)
    exp_dict[ExperimentEnum.Graphs.name] = graphs

    return exp_dict

def run_experiment():
    const_features = [ic.AROUSAL, ic.VALENCE] 
    features = [ic.AUDIO]#[ic.AUDIO, ic.VIDEO, ic.ECG, ic.EDA, ic.OTHER]

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

if __name__ == "__main__":

    path = create_experiment_folder_path()

    res = run_experiment()

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

            participant_values.append(sum(tmp_edge_histogram.values()) / len(tmp_edge_histogram))

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

        plt.show()
        #plt.savefig(os.path.join(path, f'{feat}_histogram.png'))
        plt.close()

        edge_histogram = {k: v for k, v in edge_histogram.items() if v >= max_edge_count / 2}
        ic.draw_graph(edge_histogram)
