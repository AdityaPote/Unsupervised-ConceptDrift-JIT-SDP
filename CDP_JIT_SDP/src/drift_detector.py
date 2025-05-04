import numpy as np
import torch
from sklearn.decomposition import PCA
import scipy.linalg
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import json

class DriftDetector:
    """
    Concept Drift Detector for CDP_JIT_SDP embeddings.
    
    This class implements the drift detection approach based on the Drift-Lens methodology
    to monitor and detect distribution shifts in the embedding space of the CDP_JIT_SDP model.
    """
    
    def __init__(self, n_components_batch=150, n_components_label=75):
        """
        Initialize the drift detector.
        
        Args:
            n_components_batch: Number of PCA components for batch-level embeddings
            n_components_label: Number of PCA components for label-level embeddings
        """
        self.n_components_batch = n_components_batch
        self.n_components_label = n_components_label
        
        # Reference distribution parameters
        self.reference_batch_mean = None
        self.reference_batch_cov = None
        self.reference_batch_pca = None
        
        # Label-wise reference distribution parameters
        self.reference_label_mean = {}
        self.reference_label_cov = {}
        self.reference_label_pca = {}
        
        # Threshold values
        self.batch_threshold = None
        self.label_thresholds = {}
        
        # Distribution of distances for threshold estimation
        self.batch_distances = []
        self.label_distances = defaultdict(list)
        
    def compute_statistics(self, embeddings):
        """
        Compute mean and covariance matrix for embeddings.
        
        Args:
            embeddings: Embedding vectors with shape [N, D]
            
        Returns:
            Tuple of (mean, covariance)
        """
        # Calculate mean vector
        mean = np.mean(embeddings, axis=0)
        
        # Calculate covariance matrix
        cov = np.cov(embeddings, rowvar=False)
        
        return mean, cov
    
    def estimate_baseline(self, embeddings, labels, label_list):
        """
        Estimate the baseline reference distributions from training embeddings.
        
        Args:
            embeddings: Embedding vectors with shape [N, D]
            labels: Labels with shape [N]
            label_list: List of unique labels
            
        Returns:
            Self for chaining
        """
        print("Estimating baseline distributions...")
        
        # Fit batch-level PCA
        self.reference_batch_pca = PCA(n_components=self.n_components_batch)
        batch_embeddings_reduced = self.reference_batch_pca.fit_transform(embeddings)
        
        # Compute batch-level statistics
        self.reference_batch_mean, self.reference_batch_cov = self.compute_statistics(batch_embeddings_reduced)
        
        # Compute label-wise statistics
        for label in label_list:
            # Select embeddings with this label
            label_mask = (labels == label)
            label_embeddings = embeddings[label_mask]
            
            if len(label_embeddings) > 0:
                # Fit label-level PCA
                self.reference_label_pca[label] = PCA(n_components=min(self.n_components_label, len(label_embeddings)))
                label_embeddings_reduced = self.reference_label_pca[label].fit_transform(label_embeddings)
                
                # Compute label-level statistics
                self.reference_label_mean[label], self.reference_label_cov[label] = self.compute_statistics(label_embeddings_reduced)
            else:
                print(f"Warning: No samples for label {label}")
        
        print("Baseline estimation completed")
        return self
    
    def frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate the Frechet distance between two multivariate Gaussians.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            
        Returns:
            Frechet distance
        """
        # Calculate square root of product of covariance matrices
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check for numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate Frechet distance
        fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)
        
        return fid
    
    def estimate_thresholds(self, validation_embeddings, validation_labels, window_size, n_samples=1000, percentile=95):
        """
        Estimate threshold values for drift detection from validation data.
        
        Args:
            validation_embeddings: Validation embeddings with shape [N, D]
            validation_labels: Validation labels with shape [N]
            window_size: Size of windows for distance calculation
            n_samples: Number of random windows to sample
            percentile: Percentile for threshold calculation
            
        Returns:
            Self for chaining
        """
        print(f"Estimating thresholds with {n_samples} windows of size {window_size}...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Get unique labels
        label_list = list(self.reference_label_mean.keys())
        
        for _ in tqdm(range(n_samples), desc="Sampling windows"):
            # Randomly sample a window
            indices = np.random.choice(len(validation_embeddings), window_size, replace=True)
            window_embeddings = validation_embeddings[indices]
            window_labels = validation_labels[indices]
            
            # Calculate batch-level distance
            batch_distance = self.calculate_batch_distance(window_embeddings)
            self.batch_distances.append(batch_distance)
            
            # Calculate label-wise distances
            for label in label_list:
                label_mask = (window_labels == label)
                label_embeddings = window_embeddings[label_mask]
                
                if len(label_embeddings) > 0:
                    label_distance = self.calculate_label_distance(label_embeddings, label)
                    self.label_distances[label].append(label_distance)
        
        # Calculate thresholds based on percentiles
        self.batch_threshold = np.percentile(self.batch_distances, percentile)
        
        for label in label_list:
            if len(self.label_distances[label]) > 0:
                self.label_thresholds[label] = np.percentile(self.label_distances[label], percentile)
            else:
                print(f"Warning: No distances for label {label}")
                self.label_thresholds[label] = float('inf')
        
        print("Threshold estimation completed:")
        print(f"Batch threshold: {self.batch_threshold}")
        for label in label_list:
            print(f"Label {label} threshold: {self.label_thresholds[label]}")
        
        return self
    
    def calculate_batch_distance(self, embeddings):
        """
        Calculate the Frechet distance between a batch of embeddings and the reference distribution.
        
        Args:
            embeddings: Embedding vectors with shape [N, D]
            
        Returns:
            Frechet distance
        """
        # Transform with the reference PCA
        embeddings_reduced = self.reference_batch_pca.transform(embeddings)
        
        # Calculate statistics
        mean, cov = self.compute_statistics(embeddings_reduced)
        
        # Calculate Frechet distance
        distance = self.frechet_distance(
            self.reference_batch_mean, self.reference_batch_cov,
            mean, cov
        )
        
        return distance
    
    def calculate_label_distance(self, embeddings, label):
        """
        Calculate the Frechet distance between embeddings of a specific label and the reference distribution.
        
        Args:
            embeddings: Embedding vectors with shape [N, D]
            label: Label index
            
        Returns:
            Frechet distance
        """
        # Transform with the reference PCA for this label
        embeddings_reduced = self.reference_label_pca[label].transform(embeddings)
        
        # Calculate statistics
        mean, cov = self.compute_statistics(embeddings_reduced)
        
        # Calculate Frechet distance
        distance = self.frechet_distance(
            self.reference_label_mean[label], self.reference_label_cov[label],
            mean, cov
        )
        
        return distance
    
    def detect_drift(self, embeddings, labels=None):
        """
        Detect drift in a batch of embeddings.
        
        Args:
            embeddings: Embedding vectors with shape [N, D]
            labels: Optional labels with shape [N] for label-wise drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        # Calculate batch-level distance
        batch_distance = self.calculate_batch_distance(embeddings)
        
        # Detect batch-level drift
        batch_drift = batch_distance > self.batch_threshold
        
        results = {
            'batch_distance': batch_distance,
            'batch_threshold': self.batch_threshold,
            'batch_drift': batch_drift
        }
        
        # If labels are provided, perform label-wise drift detection
        if labels is not None:
            label_results = {}
            
            for label in self.reference_label_mean.keys():
                # Select embeddings with this label
                label_mask = (labels == label)
                label_embeddings = embeddings[label_mask]
                
                if len(label_embeddings) > 0:
                    # Calculate label-wise distance
                    label_distance = self.calculate_label_distance(label_embeddings, label)
                    
                    # Detect label-wise drift
                    label_drift = label_distance > self.label_thresholds[label]
                    
                    label_results[label] = {
                        'distance': label_distance,
                        'threshold': self.label_thresholds[label],
                        'drift': label_drift
                    }
                else:
                    label_results[label] = {
                        'distance': None,
                        'threshold': self.label_thresholds[label],
                        'drift': None
                    }
            
            results['label_results'] = label_results
        
        return results
    
    def save(self, save_dir):
        """
        Save the drift detector to disk.
        
        Args:
            save_dir: Directory to save the detector
            
        Returns:
            Path to the saved model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a dictionary of attributes to save
        save_dict = {
            'n_components_batch': self.n_components_batch,
            'n_components_label': self.n_components_label,
            'reference_batch_mean': self.reference_batch_mean,
            'reference_batch_cov': self.reference_batch_cov,
            'reference_label_mean': self.reference_label_mean,
            'reference_label_cov': self.reference_label_cov,
            'batch_threshold': self.batch_threshold,
            'label_thresholds': self.label_thresholds,
            'batch_distances': self.batch_distances,
            'label_distances': dict(self.label_distances)
        }
        
        # Save the dictionary
        with open(os.path.join(save_dir, 'drift_detector.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Save PCA models separately
        with open(os.path.join(save_dir, 'reference_batch_pca.pkl'), 'wb') as f:
            pickle.dump(self.reference_batch_pca, f)
        
        for label, pca in self.reference_label_pca.items():
            with open(os.path.join(save_dir, f'reference_label_pca_{label}.pkl'), 'wb') as f:
                pickle.dump(pca, f)
        
        # Save metadata as JSON for easier inspection
        metadata = {
            'n_components_batch': self.n_components_batch,
            'n_components_label': self.n_components_label,
            'batch_threshold': float(self.batch_threshold) if self.batch_threshold is not None else None,
            'label_thresholds': {str(k): float(v) for k, v in self.label_thresholds.items()},
            'labels': list(self.reference_label_mean.keys())
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Drift detector saved to {save_dir}")
        return os.path.join(save_dir, 'drift_detector.pkl')
    
    def load(self, load_dir):
        """
        Load a saved drift detector from disk.
        
        Args:
            load_dir: Directory containing the saved detector
            
        Returns:
            Self for chaining
        """
        # Load the dictionary of attributes
        with open(os.path.join(load_dir, 'drift_detector.pkl'), 'rb') as f:
            save_dict = pickle.load(f)
        
        # Load attributes
        self.n_components_batch = save_dict['n_components_batch']
        self.n_components_label = save_dict['n_components_label']
        self.reference_batch_mean = save_dict['reference_batch_mean']
        self.reference_batch_cov = save_dict['reference_batch_cov']
        self.reference_label_mean = save_dict['reference_label_mean']
        self.reference_label_cov = save_dict['reference_label_cov']
        self.batch_threshold = save_dict['batch_threshold']
        self.label_thresholds = save_dict['label_thresholds']
        self.batch_distances = save_dict['batch_distances']
        self.label_distances = defaultdict(list)
        
        for label, distances in save_dict['label_distances'].items():
            self.label_distances[label] = distances
        
        # Load PCA models
        with open(os.path.join(load_dir, 'reference_batch_pca.pkl'), 'rb') as f:
            self.reference_batch_pca = pickle.load(f)
        
        self.reference_label_pca = {}
        for label in self.reference_label_mean.keys():
            try:
                with open(os.path.join(load_dir, f'reference_label_pca_{label}.pkl'), 'rb') as f:
                    self.reference_label_pca[label] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: PCA model for label {label} not found")
        
        print(f"Drift detector loaded from {load_dir}")
        return self 