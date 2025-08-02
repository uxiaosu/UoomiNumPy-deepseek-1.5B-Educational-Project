#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Generator - Pure NumPy Implementation

Provides text generation capabilities for the DeepSeek model.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
import time

from model import DeepSeekModel, ActivationFunctions
from tokenizer import DeepSeekTokenizer
from config import ModelConfig


class TextGenerator:
    """Text generation engine for DeepSeek model."""
    
    def __init__(self, model: DeepSeekModel, tokenizer: DeepSeekTokenizer, config: ModelConfig):
        """Initialize the text generator.
        
        Args:
            model: The DeepSeek model instance.
            tokenizer: The tokenizer instance.
            config: Model configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        verbose: bool = False
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            do_sample: Whether to use sampling or greedy decoding.
            repetition_penalty: Penalty for repeating tokens.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID.
            verbose: Whether to print generation progress.
            
        Returns:
            Generated text string.
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([input_ids], dtype=np.int64)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        if verbose:
            print(f"🎯 Starting text generation")
            print(f"   Input: {prompt}")
            print(f"   Max new tokens: {max_new_tokens}")
            print(f"   Temperature: {temperature}")
            print(f"   Top-p: {top_p}")
            print(f"   Top-k: {top_k}")
            print(f"   Input length: {input_ids.shape[1]} tokens")
        
        start_time = time.time()
        generated_tokens = 0
        
        # Generation loop
        for step in range(max_new_tokens):
            if verbose:
                print(f"🔄 Step {step + 1}: Starting forward pass...")
            
            # Forward pass
            logits = self.model.forward(input_ids)
            
            if verbose:
                print(f"✅ Step {step + 1}: Forward pass completed, logits shape: {logits.shape}")
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            
            if verbose:
                print(f"🎯 Step {step + 1}: Processing token logits...")
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids[0], repetition_penalty
                )
            
            # Sample next token
            if do_sample:
                next_token = self._sample_token(
                    next_token_logits, temperature, top_p, top_k
                )
            else:
                next_token = np.argmax(next_token_logits)
            
            if verbose:
                print(f"🎲 Step {step + 1}: Sampled token: {next_token}")
            
            # Add the new token
            input_ids = np.concatenate([
                input_ids, 
                np.array([[next_token]], dtype=np.int64)
            ], axis=1)
            
            generated_tokens += 1
            
            # Check for EOS token
            if eos_token_id is not None and next_token == eos_token_id:
                if verbose:
                    print(f"   EOS token encountered at step {step + 1}")
                break
            
            # Progress update
            if verbose and (step + 1) % 5 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
                print(f"   Generation progress: {step + 1}/{max_new_tokens} ({tokens_per_sec:.1f} tokens/s)")
        
        # Decode the generated sequence
        generated_sequence = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Extract only the newly generated part
        original_length = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        new_tokens = generated_sequence[original_length:]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if verbose:
            elapsed = time.time() - start_time
            tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
            print(f"\n✅ Generation completed!")
            print(f"   Generated tokens: {generated_tokens}")
            print(f"   Time elapsed: {elapsed:.2f}s")
            print(f"   Speed: {tokens_per_sec:.1f} tokens/s")
            print(f"   Prompt: {prompt}")
            print(f"   Generated: {new_text}")
        
        return new_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        verbose: bool = False
    ) -> str:
        """Generate a chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            verbose: Whether to print generation progress.
            
        Returns:
            Generated response string.
        """
        # Format messages using chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        if verbose:
            print(f"💬 Starting chat generation")
            print(f"------------------------------")
            print(f"Formatted input:")
            print(formatted_prompt)
        
        # Generate response
        response = self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            verbose=verbose
        )
        
        return response.strip()
    
    def _apply_repetition_penalty(
        self, 
        logits: np.ndarray, 
        input_ids: np.ndarray, 
        penalty: float
    ) -> np.ndarray:
        """Apply repetition penalty to logits."""
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
    
    def _sample_token(
        self, 
        logits: np.ndarray, 
        temperature: float, 
        top_p: float, 
        top_k: int
    ) -> int:
        """Sample next token from logits."""
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
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        verbose: bool = False
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            verbose: Whether to print generation progress.
            
        Returns:
            List of generated text strings.
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"\n🔄 Processing prompt {i + 1}/{len(prompts)}")
            
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                verbose=verbose
            )
            
            results.append(result)
        
        return results
    
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text.
        
        Args:
            text: Input text to evaluate.
            
        Returns:
            Perplexity score.
        """
        # Encode text
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = np.array([input_ids], dtype=np.int64)
        
        # Forward pass
        logits = self.model.forward(input_ids)
        
        # Calculate cross-entropy loss
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(1, input_ids.shape[1]):
            target_token = input_ids[0, i]
            token_logits = logits[0, i - 1, :]
            
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