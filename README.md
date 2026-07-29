Readme · MD
# Circuit Component Classifier
 
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)
 
A CNN that classifies hand-drawn electronic circuit components — resistors,
capacitors, AC sources, and more — from images, across multiple rotations of
each symbol.
 
## Overview
 
Hand-drawn circuit diagrams (e.g. sketched on paper or a tablet) aren't
directly usable by circuit-simulation or schematic software. The first step
toward automating that conversion is recognizing individual components. This
project trains a custom CNN to classify cropped component images into their
correct class, robust to rotation and drawing-style noise.
 
- **Task:** multi-class image classification
- **Input:** grayscale image of a single hand-drawn component
- **Output:** predicted component class + confidence
- **Approach:** custom 3-block CNN (batch norm, dropout, L2-regularized dense
  head) trained on an augmented, class-balanced image set
## Results
 
| Metric | Value |
|---|---|
| Validation accuracy | ~86%+ |
| Number of classes | 37 |
| Training images | `TBD` |
 
The model recognized most components with high confidence; minor
misclassifications occurred between symbols with visually similar
structures. Evaluated using accuracy, precision, recall, and F1-score.
 
Sample outputs to add here once available:
- Training/validation accuracy & loss curves (`plot_training_history`)
- Confusion matrix (`evaluate.py`)
- A few example predictions with confidence scores
Run `python -m circuit_classifier.evaluate` and drop the generated plots into
an `assets/` folder, then reference them here, e.g.:
`![confusion matrix](assets/confusion_matrix.png)`
 
## Project structure
 
```
circuit-component-classifier/
├── configs/
│   └── config.py            # paths + hyperparameters
├── src/circuit_classifier/
│   ├── data.py               # dataset exploration & organization
│   ├── preprocessing.py      # image preprocessing pipelines
│   ├── generators.py         # Keras data generators / augmentation
│   ├── model.py               # CNN architecture
│   ├── train.py               # training entry point (CLI)
│   ├── evaluate.py            # confusion matrix + classification report
│   ├── predict.py             # single/batch image inference (CLI)
│   └── visualize.py           # plotting helpers
├── notebooks/
│   └── demo.ipynb             # short exploratory/demo notebook
├── models/                    # trained model artifacts
├── data/                      # dataset (not tracked in git, see below)
├── tests/                     # unit tests (pytest)
└── requirements.txt
```
 
## Setup
 
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Dataset
 
This project uses the [Hand Drawn Circuit Elements dataset on Kaggle](https://www.kaggle.com/).
Download it and place it at `data/raw/dataset_final/`, with one subfolder per
class named like `<component>_r<rotation>` (e.g. `ac_src_r0`), each
containing that class's images. The dataset itself isn't tracked in git —
download it separately.
 
## Usage
 
**Train:**
 
```bash
PYTHONPATH=src python -m circuit_classifier.train \
    --dataset-path data/raw/dataset_final \
    --epochs 50
```
 
Saves the best checkpoint to `models/best_circuit_model.h5`, the final model
to `models/circuit_classifier.keras`, and metadata (class names, config,
performance) to `models/model_metadata.json`.
 
**Evaluate:**
 
```bash
PYTHONPATH=src python -m circuit_classifier.evaluate \
    --model-path models/best_circuit_model.h5
```
 
**Predict on a single image or folder:**
 
```bash
PYTHONPATH=src python -m circuit_classifier.predict --image path/to/image.png
PYTHONPATH=src python -m circuit_classifier.predict --image-dir path/to/folder
```
 
## Model architecture
 
A custom CNN with three convolutional blocks (32 → 64 → 128 filters), batch
normalization, dropout, and an L2-regularized dense head, ending in a softmax
over the component classes. See `src/circuit_classifier/model.py`.
 
## Tests
 
```bash
PYTHONPATH=src pytest tests/
```
 
## Future work
 
- Log experiments (e.g. Weights & Biases or MLflow) instead of only local
  checkpoints
- Try transfer learning from a lightweight pretrained backbone as a baseline
  to compare against the custom CNN
- Add a small Streamlit/Gradio demo for interactive single-image predictions
- Expand test coverage to the data-generator and evaluation code paths
 
