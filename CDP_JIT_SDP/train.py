from src.model import CDP_JIT_SDP
from src.tokenization import SubwordTokenizer
from src.drift_detector import DriftDetector
import torch
from tqdm import tqdm
from src.utils import save
import torch.nn as nn
import os
import datetime
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="CDP_JIT_SDP Training Script")
    
    # Data arguments
    parser.add_argument('-train_data', type=str, required=True, help='The path to the training data')
    parser.add_argument('-val_data', type=str, help='The path to the validation data')
    parser.add_argument('-dictionary_data', type=str, required=True, help='The path to dictionary data')
    
    # Training arguments
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    
    # Model architecture arguments
    parser.add_argument('-embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('-filter_sizes', type=str, default='1,2,3', help='Filter sizes for convolution')
    parser.add_argument('-num_filters', type=int, default=64, help='Number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-num_attention_heads', type=int, default=8, help='Number of attention heads')
    
    # Input formatting arguments
    parser.add_argument('-msg_length', type=int, default=256, help='Maximum message length')
    parser.add_argument('-code_line', type=int, default=10, help='Maximum number of code lines')
    parser.add_argument('-code_length', type=int, default=512, help='Maximum code line length')
    
    # Output arguments
    parser.add_argument('-save_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('-model_name', type=str, default='CDP_JIT_SDP', help='Name of the model file')
    
    # Tokenization arguments
    parser.add_argument('-use_subword_tokenization', action='store_true', help='Use subword tokenization')
    parser.add_argument('-train_tokenizers', action='store_true', help='Train new tokenizers')
    parser.add_argument('-tokenizers_dir', type=str, default='tokenizers', help='Directory for tokenizers')
    
    # Drift detection arguments
    parser.add_argument('-enable_drift_detection', action='store_true', help='Enable drift detection')
    parser.add_argument('-batch_n_pc', type=int, default=150, help='Number of principal components for batch PCA')
    parser.add_argument('-per_label_n_pc', type=int, default=75, help='Number of principal components for label PCA')
    parser.add_argument('-drift_window_size', type=int, default=1000, help='Window size for drift detection')
    parser.add_argument('-n_threshold_samples', type=int, default=1000, help='Number of samples for threshold estimation')
    
    # Device arguments
    parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('-no_cuda', action='store_true', help='Disable CUDA')
    
    return parser.parse_args()

def extract_embeddings(model, data_loader, device):
    """
    Extract embeddings from the model for all samples in the data loader.
    
    Returns:
        Tuple of (embeddings, predictions, true_labels)
    """
    model.eval()
    embeddings = []
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for msgs, codes, labels in tqdm(data_loader, desc="Extracting embeddings"):
            msgs = msgs.to(device)
            codes = codes.to(device)
            labels = labels.to(device)
            
            # Extract embeddings from the model
            batch_embeddings = model.get_embeddings(msgs, codes)
            
            # Get predictions
            outputs = model(msgs, codes)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()
            
            # Store results
            embeddings.append(batch_embeddings.cpu().numpy())
            predictions.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    
    # Concatenate results
    embeddings = np.vstack(embeddings)
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    
    return embeddings, predictions, true_labels

def create_data_loader(messages, codes, labels, batch_size, shuffle=True):
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

def train_model(args):
    """
    Train the CDP_JIT_SDP model.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device(args.device if not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.train_data}")
    with open(args.train_data, 'rb') as f:
        ids, labels, msgs, codes = pickle.load(f)
    
    # Convert labels to numpy array if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Process input data
    if args.use_subword_tokenization:
        # Use subword tokenization
        print("Using subword tokenization")
        tokenizer = SubwordTokenizer()
        
        if args.train_tokenizers:
            # Train tokenizers on the data
            print("Training tokenizers")
            tokenizer.train_tokenizers(msgs, codes)
            tokenizer.save_tokenizers(args.tokenizers_dir)
        else:
            # Load pre-trained tokenizers
            print("Loading pre-trained tokenizers")
            tokenizer.load_tokenizers(args.tokenizers_dir)
        
        # Process messages and code
        pad_msg = tokenizer.process_messages(msgs, args.msg_length)
        pad_code = tokenizer.process_code(codes, args.code_line, args.code_length)
        
        # Get vocabulary sizes
        vocab_msg, vocab_code = tokenizer.get_vocab_sizes()
        print(f"Message vocabulary size: {vocab_msg}")
        print(f"Code vocabulary size: {vocab_code}")
    else:
        # Use the original tokenization from CDP_JIT_SDP
        print("Using original tokenization")
        with open(args.dictionary_data, 'rb') as f:
            dict_msg, dict_code = pickle.load(f)
        
        from src.padding import padding_data
        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=args, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=args, type='code')
        
        vocab_msg, vocab_code = len(dict_msg), len(dict_code)
    
    # Create data loaders
    train_loader = create_data_loader(pad_msg, pad_code, labels, args.batch_size)
    
    # Set up model parameters
    args.vocab_msg = vocab_msg
    args.vocab_code = vocab_code
    args.filter_sizes = [int(k) for k in args.filter_sizes.split(',')]
    args.class_num = 1  # Binary classification
    
    # Create timestamp for saving
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create and train the model
    print("Creating model")
    model = CDP_JIT_SDP(args)
    model.to(device)
    
    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up loss function
    criterion = nn.BCELoss()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    best_loss = float('inf')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    # Early stopping
    patience = 5
    early_stopping_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        # Progress bar for batch iterations
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, (msgs, codes, lbls) in enumerate(progress_bar):
            # Move data to device
            msgs = msgs.to(device)
            codes = codes.to(device)
            lbls = lbls.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(msgs, codes)
            
            # Use weighted loss for imbalanced data
            pos_weight = torch.tensor([5.0]).to(device)  # Higher weight for positive examples
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, lbls)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} - Average loss: {avg_loss:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save model checkpoint
        save(model, save_dir, 'epoch', epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model_name}_best.pt"))
            print(f"Saved best model with loss {best_loss:.6f}")
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model_name}_final.pt"))
    print(f"Training completed. Final model saved to {save_dir}")
    
    # Set up drift detection if enabled
    if args.enable_drift_detection and args.val_data:
        print("Setting up drift detection")
        
        # Load validation data for drift detection
        with open(args.val_data, 'rb') as f:
            val_ids, val_labels, val_msgs, val_codes = pickle.load(f)
        
        # Process validation data
        if args.use_subword_tokenization:
            val_pad_msg = tokenizer.process_messages(val_msgs, args.msg_length)
            val_pad_code = tokenizer.process_code(val_codes, args.code_line, args.code_length)
        else:
            from src.padding import padding_data
            val_pad_msg = padding_data(data=val_msgs, dictionary=dict_msg, params=args, type='msg')
            val_pad_code = padding_data(data=val_codes, dictionary=dict_code, params=args, type='code')
        
        # Create validation data loader
        val_loader = create_data_loader(val_pad_msg, val_pad_code, val_labels, args.batch_size, shuffle=False)
        
        # Extract embeddings
        print("Extracting embeddings for drift detection")
        train_embeddings, train_predictions, _ = extract_embeddings(model, train_loader, device)
        val_embeddings, val_predictions, _ = extract_embeddings(model, val_loader, device)
        
        # Initialize and train drift detector
        print("Training drift detector")
        drift_detector = DriftDetector(
            n_components_batch=args.batch_n_pc,
            n_components_label=args.per_label_n_pc
        )
        
        # Estimate baseline from training data
        label_list = [0, 1]  # Binary classification
        drift_detector.estimate_baseline(
            embeddings=train_embeddings,
            labels=train_predictions,
            label_list=label_list
        )
        
        # Estimate thresholds from validation data
        drift_detector.estimate_thresholds(
            validation_embeddings=val_embeddings,
            validation_labels=val_predictions,
            window_size=args.drift_window_size,
            n_samples=args.n_threshold_samples
        )
        
        # Save drift detector
        drift_detector.save(os.path.join(save_dir, 'drift_detector'))
    
    return model, save_dir

if __name__ == "__main__":
    args = parse_arguments()
    model, save_dir = train_model(args) 