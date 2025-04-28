# ERICA Causal Experiments

This repository contains tools and scripts for causal analysis and emotion modeling using datasets such as RECOLA and SEWA. The project focuses on preprocessing, standardization, causal analysis, and emotion-link experiments to study the relationships between various features and emotional states.

## Purpose

The primary goal of this project is to:
- Perform causal analysis on emotion-related datasets.
- Standardize and preprocess raw data for consistency.
- Model and analyze the relationships between features and emotional states.
- Conduct experiments to measure the causal links between features and emotions.
- Provide a unified data structure for working with diverse datasets.

## Unified Data Structure

This project uses a common, generated data structure to perform experiments in a data-structure-agnostic context. The unified structure allows you to work with participant measures (e.g., audio, visual, physiology) and labels in a consistent way across datasets. By adding a new preprocessing script, you can easily integrate additional datasets into the workflow.

For example:
- `sewa_preprocess.py` transforms the SEWA dataset into the unified structure.
- `recola_preprocess.py` transforms the RECOLA dataset into the unified structure.

This approach ensures that all datasets are compatible with the same experimental and analytical pipelines.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd ERICA_causal_experiments
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Install Graphviz (required for visualization):
   - Download and install Graphviz from [https://graphviz.org/download/](https://graphviz.org/download/).

## Usage

### Preprocessing Data
- Use `recola_preprocess.py` to preprocess the RECOLA dataset:
  ```sh
  python recola_preprocess.py
  ```
- Use `sewa_preprocess.py` to preprocess the SEWA dataset:
  ```sh
  python sewa_preprocess.py
  ```

To add a new dataset, create a preprocessing script that transforms the raw data into the unified structure. Follow the examples in `recola_preprocess.py` and `sewa_preprocess.py`.

### Standardizing Data
- Run `standardization.py` to standardize the preprocessed data:
  ```sh
  python standardization.py
  ```

### Emotion-Link Experiments
- Run experiments to measure emotion links using the scripts in the `emotion_link_tests/` directory:
  ```sh
  python emotion_link_tests/experiment_1.py
  ```

### Integrity Checks
- Use `integrity_check.py` to verify the integrity of the data:
  ```sh
  python integrity_check.py
  ```

### Annotation Range Check
- Run `check_annotations_range.py` to validate annotation ranges:
  ```sh
  python check_annotations_range.py
  ```

## Project Structure

```
ERICA_causal_experiments/
├── data/                     # Contains raw, preprocessed, and standardized datasets
│   ├── preprocessed/         # Preprocessed data
│   ├── RECOLA-DATA/          # Raw RECOLA dataset
│   ├── standardized/         # Standardized data
├── emotion_link_tests/       # Scripts for emotion-link experiments
│   ├── experiment_1.py       # Experiment 1 script
│   ├── measure_emotion_link.py # Utility for measuring emotion links
├── causal_analysis_recola.ipynb # Jupyter notebook for causal analysis
├── recola_preprocess.py      # Preprocessing script for RECOLA dataset
├── sewa_preprocess.py        # Preprocessing script for SEWA dataset
├── standardization.py        # Standardization script
├── integrity_check.py        # Data integrity check script
├── check_annotations_range.py # Annotation range validation script
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

## Basic Functionality

1. **Preprocessing**: Prepares raw datasets for analysis by cleaning and formatting the data into a unified structure.
2. **Standardization**: Ensures consistency across datasets by applying standard scaling and transformations.
3. **Causal Analysis**: Identifies causal relationships between features and emotional states using statistical methods.
4. **Emotion-Link Experiments**: Tests hypotheses about the causal links between features and emotions.

## Dependencies

- Python 3.x
- Graphviz
- Libraries listed in `requirements.txt`

## License

This project is licensed under the MIT License. See the LICENSE file for details.