import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
from tqdm import tqdm

class SubwordTokenizer:
    def __init__(self, vocab_size=50000, min_frequency=2):
        """
        Initialize subword tokenizer for code and commit messages.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a token to be included
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.code_tokenizer = None
        self.msg_tokenizer = None
        
    def train_tokenizers(self, msg_data: List[str], code_data: List[List[str]]):
        """
        Train separate tokenizers for commit messages and code.
        
        Args:
            msg_data: List of commit messages
            code_data: List of code changes (flattened)
        """
        print("Training message tokenizer...")
        self.msg_tokenizer = self._train_tokenizer(msg_data, "msg_tokenizer")
        
        # Flatten code data for training
        flattened_code = []
        for commit in code_data:
            for line in commit:
                flattened_code.append(line)
        
        print("Training code tokenizer...")
        self.code_tokenizer = self._train_tokenizer(flattened_code, "code_tokenizer")
        
    def _train_tokenizer(self, data: List[str], name: str) -> PreTrainedTokenizerFast:
        """
        Train a BPE tokenizer on the provided data.
        
        Args:
            data: List of text samples
            name: Name of the tokenizer
            
        Returns:
            Trained tokenizer
        """
        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Configure special tokens and post-processing
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Initialize a BPE trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<NULL>"],
        )
        
        # Prepare the training corpus with progress bar
        corpus = []
        for text in tqdm(data, desc=f"Preparing {name} corpus"):
            # Clean the text
            if isinstance(text, str):
                text = text.strip().lower()
                if text:  # Only add non-empty texts
                    corpus.append(text)
        
        # Train the tokenizer
        tokenizer.train_from_iterator(corpus, trainer)
        
        # Save the tokenizer
        os.makedirs("tokenizers", exist_ok=True)
        tokenizer.save(f"tokenizers/{name}.json")
        
        # Convert to PreTrainedTokenizerFast for easier use
        pretrained_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"tokenizers/{name}.json",
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token=None
        )
        
        return pretrained_tokenizer
    
    def save_tokenizers(self, save_dir: str):
        """
        Save the trained tokenizers.
        
        Args:
            save_dir: Directory to save tokenizers
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.msg_tokenizer:
            self.msg_tokenizer.save_pretrained(os.path.join(save_dir, "msg_tokenizer"))
        
        if self.code_tokenizer:
            self.code_tokenizer.save_pretrained(os.path.join(save_dir, "code_tokenizer"))
    
    def load_tokenizers(self, load_dir: str):
        """
        Load previously trained tokenizers.
        
        Args:
            load_dir: Directory containing saved tokenizers
        """
        self.msg_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(load_dir, "msg_tokenizer")
        )
        
        self.code_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(load_dir, "code_tokenizer")
        )
    
    def encode_message(self, message: str, max_length: int) -> np.ndarray:
        """
        Encode a commit message.
        
        Args:
            message: Commit message
            max_length: Maximum sequence length
            
        Returns:
            Tensor of token IDs
        """
        if not self.msg_tokenizer:
            raise ValueError("Message tokenizer not initialized. Train or load tokenizers first.")
        
        encoded = self.msg_tokenizer.encode(
            message.lower().strip(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        
        return encoded.numpy()[0]
    
    def encode_code(self, code_lines: List[str], max_lines: int, max_length: int) -> np.ndarray:
        """
        Encode code lines.
        
        Args:
            code_lines: List of code lines
            max_lines: Maximum number of lines
            max_length: Maximum length per line
            
        Returns:
            Tensor of token IDs with shape [max_lines, max_length]
        """
        if not self.code_tokenizer:
            raise ValueError("Code tokenizer not initialized. Train or load tokenizers first.")
        
        # Prepare code lines
        if len(code_lines) > max_lines:
            # Truncate if too many lines
            code_lines = code_lines[:max_lines]
        elif len(code_lines) < max_lines:
            # Pad with empty lines if too few
            code_lines = code_lines + ["<NULL>"] * (max_lines - len(code_lines))
        
        # Encode each line
        encoded_lines = []
        for line in code_lines:
            encoded = self.code_tokenizer.encode(
                line.lower().strip(),
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            encoded_lines.append(encoded.numpy()[0])
        
        return np.array(encoded_lines)
    
    def process_messages(self, messages: List[str], max_length: int) -> np.ndarray:
        """
        Process a batch of commit messages.
        
        Args:
            messages: List of commit messages
            max_length: Maximum sequence length
            
        Returns:
            Array of token IDs for all messages
        """
        encoded_msgs = []
        for msg in tqdm(messages, desc="Processing messages"):
            encoded_msgs.append(self.encode_message(msg, max_length))
        
        return np.array(encoded_msgs)
    
    def process_code(self, code_data: List[List[str]], max_lines: int, max_length: int) -> np.ndarray:
        """
        Process a batch of code changes.
        
        Args:
            code_data: List of code changes (list of lines per commit)
            max_lines: Maximum number of lines
            max_length: Maximum length per line
            
        Returns:
            Array of token IDs for all code changes
        """
        encoded_code = []
        for code in tqdm(code_data, desc="Processing code"):
            encoded_code.append(self.encode_code(code, max_lines, max_length))
        
        return np.array(encoded_code)
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """Get vocabulary sizes for message and code tokenizers."""
        msg_vocab_size = len(self.msg_tokenizer.get_vocab()) if self.msg_tokenizer else 0
        code_vocab_size = len(self.code_tokenizer.get_vocab()) if self.code_tokenizer else 0
        return msg_vocab_size, code_vocab_size
    
    def prepare_dictionaries(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Prepare dictionaries compatible with CDP_JIT_SDP's existing format.
        
        Returns:
            Tuple of (message_dict, code_dict)
        """
        msg_dict = self.msg_tokenizer.get_vocab()
        code_dict = self.code_tokenizer.get_vocab()
        
        # Ensure <NULL> token is present
        if "<NULL>" not in msg_dict:
            msg_dict["<NULL>"] = len(msg_dict)
        if "<NULL>" not in code_dict:
            code_dict["<NULL>"] = len(code_dict)
            
        return msg_dict, code_dict 