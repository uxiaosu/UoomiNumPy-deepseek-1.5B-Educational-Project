#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling Utilities Module

Utilities for token sampling during text generation.
"""

import numpy as np
from typing import Optional

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model import ActivationFunctions


class SamplingUtils:
    """Utilities for sampling tokens during generation."""
    
    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray, 
        input_ids: np.ndarray, 
        penalty: float
    ) -> np.ndarray:
        """Apply repetition penalty to logits.
        
        Args:
            logits: Token logits
            input_ids: Previously generated token IDs
            penalty: Repetition penalty factor
            
        Returns:
            Penalized logits
        """
        if penalty == 1.0:
            return logits
        
        # Create a copy of logits
        penalized_logits = logits.copy()
        
        # Apply penalty to tokens that have appeared before
        unique_tokens = np.unique(input_ids)
        for token in unique_tokens:
            if 0 <= token < len(penalized_logits):
                if penalized_logits[token] > 0:
                    penalized_logits[token] /= penalty
                else:
                    penalized_logits[token] *= penalty
        
        return penalized_logits
    
    @staticmethod
    def sample_token(
        logits: np.ndarray, 
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        top_k: int = 0
    ) -> int:
        """Sample next token from logits.
        
        Args:
            logits: Token logits
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, len(logits))
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -np.inf
        
        # Convert to probabilities
        probs = ActivationFunctions.softmax(logits)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff index
            cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1
            cutoff_index = min(cutoff_index, len(sorted_indices))
            
            # Zero out probabilities beyond cutoff
            indices_to_remove = np.ones(len(probs), dtype=bool)
            indices_to_remove[sorted_indices[:cutoff_index]] = False
            probs[indices_to_remove] = 0.0
            
            # Renormalize
            probs = probs / np.sum(probs)
        
        # Sample from the distribution
        try:
            token = np.random.choice(len(probs), p=probs)
        except ValueError:
            # Fallback to argmax if sampling fails
            token = np.argmax(probs)
        
        return int(token)
    
    @staticmethod
    def greedy_sample(logits: np.ndarray) -> int:
        """Greedy sampling (select token with highest probability).
        
        Args:
            logits: Token logits
            
        Returns:
            Token ID with highest probability
        """
        return int(np.argmax(logits))
    
    @staticmethod
    def calculate_perplexity(
        logits: np.ndarray, 
        target_ids: np.ndarray
    ) -> float:
        """Calculate perplexity for given logits and targets.
        
        Args:
            logits: Model logits of shape (seq_len, vocab_size)
            target_ids: Target token IDs of shape (seq_len,)
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(len(target_ids)):
            target_token = target_ids[i]
            token_logits = logits[i]
            
            # Apply softmax
            probs = ActivationFunctions.softmax(token_logits)
            
            # Calculate negative log likelihood
            token_loss = -np.log(probs[target_token] + 1e-10)
            total_loss += token_loss
            total_tokens += 1
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss)
        
        return float(perplexity)