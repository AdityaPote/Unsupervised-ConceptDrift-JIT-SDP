from src.model import CDP_JIT_SDP
from src.tokenization import SubwordTokenizer
from src.drift_detector import DriftDetector
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import json
import time
from src.padding import padding_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="CDP_JIT_SDP Evaluation")
    
    # Data arguments
    parser.add_argument('-test_data', type=str, required=True, help='The path to the test data')
    parser.add_argument('-dictionary_data', type=str, help='The path to dictionary data')
    
    # Model arguments
    parser.add_argument('-load_model', type=str, required=True, help='The path to the saved model')
    parser.add_argument('-drift_detector', type=str, help='Path to drift detector directory')
    
    # Input formatting arguments
    parser.add_argument('-msg_length', type=int, default=256, help='Maximum message length')
    parser.add_argument('-code_line', type=int, default=10, help='Maximum number of code lines')
    parser.add_argument('-code_length', type=int, default=512, help='Maximum code line length')
    
    # Model architecture arguments
    parser.add_argument('-embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('-filter_sizes', type=str, default='1,2,3', help='Filter sizes for convolution')
    parser.add_argument('-num_filters', type=int, default=64, help='Number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-num_attention_heads', type=int, default=8, help='Number of attention heads')
    
    # Tokenization arguments
    parser.add_argument('-use_subword_tokenization', action='store_true', help='Use subword tokenization')
    parser.add_argument('-tokenizers_dir', type=str, default='tokenizers', help='Directory for tokenizers')
    
    # Drift detection arguments
    parser.add_argument('-detect_drift', action='store_true', help='Detect drift in test data')
    parser.add_argument('-drift_window_size', type=int, default=1000, help='Window size for drift detection')
    
    # Output arguments
    parser.add_argument('-output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('-visualize', action='store_true', help='Visualize results')
    
    # Device arguments
    parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('-no_cuda', action='store_true', help='Disable CUDA')
    
    return parser.parse_args()

def create_data_loader(messages, codes, labels, batch_size=64, shuffle=False):
    """Create a PyTorch DataLoader from the input data."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(messages, dtype=torch.long),
        torch.tensor(codes, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float)
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the test data.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for msgs, codes, labels in tqdm(data_loader, desc="Evaluating"):
            msgs = msgs.to(device)
            codes = codes.to(device)
            
            # Get model output
            outputs = model(msgs, codes)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            
            # Store results
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend((probabilities > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }
    
    return metrics, all_probabilities, all_predictions, all_labels

def extract_embeddings(model, data_loader, device):
    """
    Extract embeddings from the model for all samples in the data loader.
    
    Returns:
        Tuple of (embeddings, predictions, labels)
    """
    model.eval()
    embeddings = []
    predictions = []
    labels = []
    
    with torch.no_grad():
        for msgs, codes, lbls in tqdm(data_loader, desc="Extracting embeddings"):
            msgs = msgs.to(device)
            codes = codes.to(device)
            
            # Extract embeddings
            batch_embeddings = model.get_embeddings(msgs, codes)
            
            # Get predictions
            outputs = model(msgs, codes)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()
            
            # Store results
            embeddings.append(batch_embeddings.cpu().numpy())
            predictions.append(preds.cpu().numpy())
            labels.append(lbls.numpy())
    
    # Concatenate results
    embeddings = np.vstack(embeddings)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    
    return embeddings, predictions, labels

def detect_drift_in_windows(drift_detector, embeddings, predictions, window_size):
    """
    Detect drift in windows of embeddings.
    
    Returns:
        List of drift detection results per window
    """
    n_samples = len(embeddings)
    n_windows = n_samples // window_size
    
    print(f"Detecting drift in {n_windows} windows of size {window_size}")
    
    drift_results = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, n_samples)
        
        window_embeddings = embeddings[start_idx:end_idx]
        window_predictions = predictions[start_idx:end_idx]
        
        # Detect drift in this window
        result = drift_detector.detect_drift(window_embeddings, window_predictions)
        result['window_id'] = i
        result['start_idx'] = start_idx
        result['end_idx'] = end_idx
        
        drift_results.append(result)
    
    return drift_results

def visualize_results(metrics, drift_results=None, output_dir=None):
    """
    Visualize evaluation results.
    
    Args:
        metrics: Dictionary with evaluation metrics
        drift_results: List of drift detection results per window
        output_dir: Directory to save visualizations
    """
    # Set up plotting
    plt.figure(figsize=(10, 8))
    
    # Create a bar chart of key metrics
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_values = [metrics[metric] for metric in key_metrics]
    
    plt.bar(key_metrics, metric_values, color='skyblue')
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # If drift results are available, visualize drift detection
    if drift_results:
        plt.figure(figsize=(12, 6))
        
        # Extract batch-level drift information
        window_ids = [result['window_id'] for result in drift_results]
        batch_distances = [result['batch_distance'] for result in drift_results]
        batch_threshold = drift_results[0]['batch_threshold']
        batch_drifts = [result['batch_drift'] for result in drift_results]
        
        # Plot batch distances and threshold
        plt.plot(window_ids, batch_distances, marker='o', linestyle='-', label='Batch Distance')
        plt.axhline(y=batch_threshold, color='r', linestyle='--', label='Threshold')
        
        # Highlight windows with drift
        for i, drift in enumerate(batch_drifts):
            if drift:
                plt.axvspan(window_ids[i] - 0.5, window_ids[i] + 0.5, color='red', alpha=0.3)
        
        plt.title('Drift Detection Results')
        plt.xlabel('Window ID')
        plt.ylabel('Frechet Distance')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'drift_detection.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # If label-level results are available, visualize them
        if 'label_results' in drift_results[0]:
            plt.figure(figsize=(12, 6))
            
            # Get labels
            labels = list(drift_results[0]['label_results'].keys())
            
            for label in labels:
                # Extract label distances
                label_distances = []
                for result in drift_results:
                    if result['label_results'][label]['distance'] is not None:
                        label_distances.append(result['label_results'][label]['distance'])
                    else:
                        label_distances.append(0)  # No samples for this label in the window
                
                plt.plot(window_ids[:len(label_distances)], label_distances, marker='o', linestyle='-', label=f'Label {label}')
                
                # Plot threshold for this label
                label_threshold = drift_results[0]['label_results'][label]['threshold']
                plt.axhline(y=label_threshold, linestyle='--', alpha=0.5)
            
            plt.title('Label-Level Drift Detection')
            plt.xlabel('Window ID')
            plt.ylabel('Frechet Distance')
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'label_drift_detection.png'), dpi=300, bbox_inches='tight')
            
            plt.show()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'rb') as f:
        ids, labels, msgs, codes = pickle.load(f)
    
    # Convert labels to numpy array if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Process input data
    if args.use_subword_tokenization:
        # Use subword tokenization
        print("Using subword tokenization")
        tokenizer = SubwordTokenizer()
        
        # Load pre-trained tokenizers
        print(f"Loading tokenizers from {args.tokenizers_dir}")
        tokenizer.load_tokenizers(args.tokenizers_dir)
        
        # Process messages and code
        pad_msg = tokenizer.process_messages(msgs, args.msg_length)
        pad_code = tokenizer.process_code(codes, args.code_line, args.code_length)
        
        # Get vocabulary sizes
        vocab_msg, vocab_code = tokenizer.get_vocab_sizes()
    else:
        # Use the original tokenization from CDP_JIT_SDP
        print("Using original tokenization")
        with open(args.dictionary_data, 'rb') as f:
            dict_msg, dict_code = pickle.load(f)
        
        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=args, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=args, type='code')
        
        vocab_msg, vocab_code = len(dict_msg), len(dict_code)
    
    # Create data loader
    test_loader = create_data_loader(pad_msg, pad_code, labels)
    
    # Set up model parameters
    args.vocab_msg = vocab_msg
    args.vocab_code = vocab_code
    args.filter_sizes = [int(k) for k in args.filter_sizes.split(',')]
    args.class_num = 1  # Binary classification
    
    # Create model and load weights
    print(f"Loading model from {args.load_model}")
    model = CDP_JIT_SDP(args)
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)
    
    # Evaluate model
    print("Evaluating model")
    start_time = time.time()
    metrics, probabilities, predictions, true_labels = evaluate_model(model, test_loader, device)
    evaluation_time = time.time() - start_time
    
    # Print results
    print("\nEvaluation Results:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    
    # Save metrics
    if args.output_dir:
        metrics['evaluation_time'] = evaluation_time
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Perform drift detection if requested
    if args.detect_drift and args.drift_detector:
        print("Performing drift detection")
        
        # Extract embeddings for drift detection
        embeddings, pred_labels, _ = extract_embeddings(model, test_loader, device)
        
        # Load drift detector
        drift_detector = DriftDetector()
        drift_detector.load(args.drift_detector)
        
        # Detect drift in windows
        drift_results = detect_drift_in_windows(
            drift_detector, 
            embeddings, 
            pred_labels, 
            args.drift_window_size
        )
        
        # Count windows with drift
        drift_count = sum(1 for result in drift_results if result['batch_drift'])
        print(f"Detected drift in {drift_count} out of {len(drift_results)} windows")
        
        # Save drift results
        if args.output_dir:
            # Convert numpy values to Python types for JSON serialization
            serializable_results = []
            for result in drift_results:
                serializable_result = {
                    'window_id': int(result['window_id']),
                    'start_idx': int(result['start_idx']),
                    'end_idx': int(result['end_idx']),
                    'batch_distance': float(result['batch_distance']),
                    'batch_threshold': float(result['batch_threshold']),
                    'batch_drift': bool(result['batch_drift'])
                }
                
                if 'label_results' in result:
                    serializable_result['label_results'] = {}
                    for label, label_result in result['label_results'].items():
                        serializable_result['label_results'][label] = {
                            'distance': float(label_result['distance']) if label_result['distance'] is not None else None,
                            'threshold': float(label_result['threshold']),
                            'drift': bool(label_result['drift']) if label_result['drift'] is not None else None
                        }
                
                serializable_results.append(serializable_result)
            
            with open(os.path.join(args.output_dir, 'drift_results.json'), 'w') as f:
                json.dump(serializable_results, f, indent=2)
    else:
        drift_results = None
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(metrics, drift_results, args.output_dir)

if __name__ == "__main__":
    main() 