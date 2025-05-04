# CDP_JIT_SDP: Improved Just-In-Time Defect Prediction

Populate the data folder with dataset from this link
https://zenodo.org/records/3965246#.XyEDVnUzY5k

This repository contains an enhanced version of the DeepJIT model for Just-In-Time defect prediction. The original DeepJIT model has been improved with the following enhancements:

1. **Self-Attention Mechanism**: Added both self-attention and cross-attention layers to better capture relationships between different parts of commit messages and code changes.

2. **Subword Tokenization**: Implemented subword tokenization using Byte-Pair Encoding (BPE) to better handle the vocabulary of code and comments.

3. **Concept Drift Detection**: Integrated a concept drift detection mechanism based on the Drift-Lens approach to monitor and detect when the model's performance might degrade due to changes in the code patterns.

## Components

The enhanced version consists of the following main components:

- `src/model.py`: Contains the CDP_JIT_SDP model with attention mechanisms
- `src/tokenization.py`: Implements subword tokenization for code and commit messages
- `src/drift_detector.py`: Implements concept drift detection based on Drift-Lens
- `src/utils.py`: Utility functions for processing data and saving models
- `src/padding.py`: Functions for padding input data to fixed lengths
- `train.py`: Training script for the enhanced model
- `evaluate.py`: Evaluation script with extended metrics and drift detection

## Requirements

Install the required packages:

```
pip install -r requirements.txt
```

## Usage

### Training

To train the enhanced model with all improvements:

```bash
python train.py \
    -train_data ./data/openstack_train.pkl \
    -val_data ./data/openstack_test.pkl \
    -dictionary_data ./data/openstack_dict.pkl \
    -save_dir ./models \
    -model_name CDP_JIT_SDP \
    -epochs 10 \
    -batch_size 64 \
    -learning_rate 0.001 \
    -embedding_dim 128 \
    -hidden_units 512 \
    -num_attention_heads 8 \
    -use_subword_tokenization \
    -train_tokenizers \
    -enable_drift_detection \
    -batch_n_pc 150 \
    -per_label_n_pc 75
```

To train with just attention mechanism (without subword tokenization and drift detection):

```bash
python train.py \
    -train_data ./data/openstack_train.pkl \
    -dictionary_data ./data/openstack_dict.pkl \
    -save_dir ./models \
    -model_name CDP_JIT_SDP_basic \
    -epochs 10
```

### Evaluation

To evaluate the enhanced model:

```bash
python evaluate.py \
    -test_data ./data/openstack_test.pkl \
    -dictionary_data ./data/openstack_dict.pkl \
    -load_model ./models/<timestamp>/CDP_JIT_SDP_best.pt \
    -output_dir ./results \
    -visualize
```

To evaluate with drift detection:

```bash
python evaluate.py \
    -test_data ./data/openstack_test.pkl \
    -dictionary_data ./data/openstack_dict.pkl \
    -load_model ./models/<timestamp>/CDP_JIT_SDP_best.pt \
    -drift_detector ./models/<timestamp>/drift_detector \
    -detect_drift \
    -drift_window_size 1000 \
    -output_dir ./results \
    -visualize
```

For subword tokenization, add the `-use_subword_tokenization` flag and specify the tokenizers directory:

```bash
python evaluate.py \
    -test_data ./data/openstack_test.pkl \
    -load_model ./models/<timestamp>/CDP_JIT_SDP_best.pt \
    -use_subword_tokenization \
    -tokenizers_dir ./tokenizers \
    -output_dir ./results \
    -visualize
```

## Improvements Explained

### 1. Attention Mechanism

The original DeepJIT model used Convolutional Neural Networks (CNNs) to extract features from commit messages and code changes. While CNNs are effective at capturing local patterns, they may miss long-range dependencies between different parts of the code or message.

The enhanced model adds several attention mechanisms:

- **Self-attention for messages**: Allows the model to focus on important parts of the commit message
- **Self-attention for code**: Helps the model understand relationships between different lines of code
- **Cross-attention**: Captures dependencies between code and commit messages

This architecture better represents the relationships between different parts of the commit and improves the model's ability to identify defect-prone patterns.

### 2. Subword Tokenization

The original DeepJIT model used simple word-level tokenization, which has limitations in handling out-of-vocabulary words and programming-specific tokens. This is particularly problematic for code, where identifiers, function names, and code patterns may be unique.

The enhanced model implements subword tokenization using Byte-Pair Encoding (BPE), which:

- Breaks words into subword units, allowing better handling of rare words
- Reduces the out-of-vocabulary problem
- Captures meaningful subword units in code (e.g., parts of function names, code patterns)

### 3. Concept Drift Detection

Software projects evolve over time, and the patterns of defects can change. This can lead to a degradation of the model's performance. The enhanced model includes a concept drift detection mechanism that:

- Monitors the distribution of features in the model's embeddings
- Detects when the distribution shifts significantly from the training distribution
- Provides early warning when the model might need retraining
- Can identify which specific types of commits (labels) are experiencing drift

The drift detection approach is based on the Drift-Lens methodology, using Frechet Inception Distance (FID) to compare the distributions.

## Evaluation Metrics

The enhanced evaluation provides a more comprehensive set of metrics:

- **AUC**: Area Under the ROC Curve
- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall**: Proportion of true positives that are correctly predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: Proportion of true negatives that are correctly predicted
- **False Positive Rate**: Proportion of false positives among negative samples
- **False Negative Rate**: Proportion of false negatives among positive samples

## Results

The enhanced model is expected to improve upon the original DeepJIT model in the following ways:

1. Higher AUC score due to better feature extraction with attention mechanisms
2. More robust handling of code vocabulary with subword tokenization
3. Early detection of model performance degradation with drift detection
4. Better overall defect prediction accuracy and precision

## Credits

This work builds upon the original DeepJIT model by Thong Hoang et al., incorporating concepts from Drift-Lens for concept drift detection and modern NLP techniques for better tokenization and feature extraction. 