#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Generator Module

Main text generation class using modular components.
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model import DeepSeekModel, ActivationFunctions
from tokenizer import DeepSeekTokenizer
from config import ModelConfig

# Import from current package
from .sampling_utils import SamplingUtils
from .generation_utils import GenerationUtils


class TextGenerator:
    """Text generation engine for DeepSeek model.
    
    This class provides high-level text generation capabilities including:
    - Single text generation
    - Chat-style generation
    - Batch generation
    - Perplexity calculation
    """
    
    def __init__(self, model: DeepSeekModel, tokenizer: DeepSeekTokenizer, config: ModelConfig):
        """Initialize text generator.
        
        Args:
            model: DeepSeek model instance
            tokenizer: DeepSeek tokenizer instance
            config: Model configuration
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
        # Validate parameters
        GenerationUtils.validate_generation_params(
            max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )
        
        # Prepare input
        input_ids = GenerationUtils.prepare_input_ids(self.tokenizer, prompt)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        if verbose:
            GenerationUtils.print_generation_start(
                prompt, max_new_tokens, temperature, top_p, top_k, input_ids.shape[1]
            )
        
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
                next_token_logits = SamplingUtils.apply_repetition_penalty(
                    next_token_logits, input_ids[0], repetition_penalty
                )
            
            # Sample next token
            if do_sample:
                next_token = SamplingUtils.sample_token(
                    next_token_logits, temperature, top_p, top_k
                )
            else:
                next_token = SamplingUtils.greedy_sample(next_token_logits)
            
            if verbose:
                GenerationUtils.print_generation_progress(
                    step, max_new_tokens, start_time, generated_tokens + 1, next_token
                )
            
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
        
        # Extract generated text
        generated_sequence = input_ids[0].tolist()
        new_text = GenerationUtils.extract_new_tokens(
            self.tokenizer, prompt, generated_sequence
        )
        
        if verbose:
            GenerationUtils.print_generation_summary(
                prompt, new_text, generated_tokens, start_time, temperature, top_p, top_k
            )
        
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
        formatted_prompt = GenerationUtils.format_chat_messages(
            self.tokenizer, messages
        )
        
        if verbose:
            GenerationUtils.print_chat_start(formatted_prompt)
        
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
        # Prepare input
        input_ids = GenerationUtils.prepare_input_ids(self.tokenizer, text)
        
        # Forward pass
        logits = self.model.forward(input_ids)
        
        # Calculate perplexity for the sequence (excluding first token)
        if input_ids.shape[1] <= 1:
            return float('inf')
        
        target_ids = input_ids[0, 1:]  # Skip first token
        sequence_logits = logits[0, :-1, :]  # Skip last logit
        
        perplexity = SamplingUtils.calculate_perplexity(sequence_logits, target_ids)
        
        return perplexity