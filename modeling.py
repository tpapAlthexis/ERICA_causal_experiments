import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score

import integrity_check
import data_acquisition as da
import globals as gl
import independence_causal as ic

DATASET = gl.Dataset.RECOLA
MEASURES = [gl.AUDIO, gl.VIDEO]

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
   

if __name__ == "__main__":
    p_to_avoid = integrity_check.is_ready_for_experiment(DATASET)
    if p_to_avoid:
        print(f'Experiment will be run excluding certain participants. Total participants to avoid:{len(p_to_avoid)}') 

    participants = gl.getParticipants(DATASET)
    participants = [p for p in participants if p not in p_to_avoid]

    # Split the participants into training and test sets (20% of participants will be used for testing)
    train_participants, test_participants = train_test_split(participants, test_size=0.2, shuffle=False)

    print(f"Training participants: {train_participants}")
    print(f"Test participants: {test_participants}")

    train_features, train_targets = read_data(p_to_avoid=test_participants, apply_ica=False)
    if train_features is None or train_targets is None:
        print("Data and/or annotations are empty")
        exit()
    if train_features.shape[0] != train_targets.shape[0]:
        print("Data and annotations rows are not equal")
        exit()

    test_features, test_targets = read_data(p_to_avoid=train_participants, apply_ica=False)
    if test_features is None or test_targets is None:
        print("Data and/or annotations are empty")
        exit()
    if test_features.shape[0] != test_targets.shape[0]:
        print("Data and annotations rows are not equal")
        exit()

    train_valence = train_targets.filter(like=gl.AROUSAL, axis=1)
    test_valence = test_targets.filter(like=gl.AROUSAL, axis=1)

    model_valence = SVR()
    model_valence.fit(train_features, np.ravel(train_valence))

    # Predicting the test set results
    y_pred_valence = model_valence.predict(test_features)
    print('Valence Model Test MSE:', mean_squared_error(test_valence, y_pred_valence))
    print('Valence Model Test R^2:', r2_score(test_valence, y_pred_valence))
        

    


    
    

    