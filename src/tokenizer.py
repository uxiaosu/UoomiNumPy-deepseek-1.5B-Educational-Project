#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Tokenizer - Pure Python Implementation

A complete tokenizer implementation for DeepSeek models without external dependencies.
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


class DeepSeekTokenizer:
    """Pure Python tokenizer implementation for DeepSeek models."""
    
    def __init__(self, tokenizer_dir: str):
        """Initialize the tokenizer.
        
        Args:
            tokenizer_dir: Path to the tokenizer directory containing tokenizer files.
        """
        self.tokenizer_dir = Path(tokenizer_dir)
        
        # Core tokenizer components
        self.vocab = {}  # token -> id
        self.id_to_token = {}  # id -> token
        self.merges = []  # BPE merge rules
        self.special_tokens = {}
        
        # Configuration
        self.config = {}
        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.pad_token = None
        
        # Load tokenizer files
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer from files."""
        # Load tokenizer.json
        tokenizer_json_path = self.tokenizer_dir / "tokenizer.json"
        if tokenizer_json_path.exists():
            self._load_tokenizer_json(tokenizer_json_path)
        
        # Load tokenizer_config.json
        config_path = self.tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            self._load_tokenizer_config(config_path)
    
    def _load_tokenizer_json(self, tokenizer_json_path: Path):
        """Load tokenizer.json file."""
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Load vocabulary
        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
            self.vocab = tokenizer_data['model']['vocab']
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load BPE merge rules
        if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
            self.merges = tokenizer_data['model']['merges']
        
        # Load special tokens
        if 'added_tokens' in tokenizer_data:
            for token_info in tokenizer_data['added_tokens']:
                content = token_info['content']
                token_id = token_info['id']
                self.special_tokens[content] = token_id
    
    def _load_tokenizer_config(self, config_path: Path):
        """Load tokenizer_config.json file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Set special tokens
        def extract_token(token_config, default):
            if isinstance(token_config, dict):
                return token_config.get('content', default)
            return token_config or default
        
        self.bos_token = extract_token(self.config.get('bos_token'), '<|begin_of_text|>')
        self.eos_token = extract_token(self.config.get('eos_token'), '<|end_of_text|>')
        self.unk_token = extract_token(self.config.get('unk_token'), '<|reserved_0|>')
        self.pad_token = extract_token(self.config.get('pad_token'), '<|end_of_text|>')
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get character pairs from word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe_encode(self, text: str) -> List[str]:
        """Apply BPE encoding to text."""
        if not text:
            return []
        
        # Preprocess text to handle spaces properly
        # Replace spaces with a special marker to preserve them
        text = text.replace(' ', 'Ġ')
        
        # Convert text to character list
        word_tokens = list(text)
        
        # Apply BPE merge rules
        while len(word_tokens) > 1:
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break
            
            # Find highest priority merge
            bigram = None
            min_merge_idx = float('inf')
            
            for pair in pairs:
                merge_str = f"{pair[0]} {pair[1]}"
                if merge_str in self.merges:
                    merge_idx = self.merges.index(merge_str)
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        bigram = pair
            
            if bigram is None:
                break
            
            # Perform merge
            new_word = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == bigram[0] and 
                    word_tokens[i + 1] == bigram[1]):
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_word
        
        return word_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add special tokens.
            
        Returns:
            List of token IDs.
        """
        if not text:
            return []
        
        # Preprocess text
        text = text.strip()
        
        # BPE encoding
        tokens = self._bpe_encode(text)
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            elif token in self.special_tokens:
                token_ids.append(self.special_tokens[token])
            else:
                # Unknown token
                if self.unk_token and self.unk_token in self.vocab:
                    token_ids.append(self.vocab[self.unk_token])
        
        # Add special tokens
        if add_special_tokens:
            if self.bos_token and self.bos_token in self.vocab:
                token_ids.insert(0, self.vocab[self.bos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.
            
        Returns:
            Decoded text string.
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and self._is_special_token(token):
                    continue
                
                tokens.append(token)
        
        # Merge tokens to text
        text = self._merge_tokens(tokens)
        return text
    
    def _is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        special_tokens = [self.bos_token, self.eos_token, self.unk_token, self.pad_token]
        return token in special_tokens or token in self.special_tokens
    
    def _merge_tokens(self, tokens: List[str]) -> str:
        """Merge tokens into text."""
        if not tokens:
            return ""
        
        # Simple merge strategy
        text = "".join(tokens)
        
        # Handle spaces - restore from space markers
        text = re.sub(r'Ġ', ' ', text)  # GPT-2 style space marker
        text = re.sub(r'▁', ' ', text)  # SentencePiece style space marker
        
        # Clean up multiple spaces and trim
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get special tokens dictionary."""
        special_dict = {}
        
        if self.bos_token and self.bos_token in self.vocab:
            special_dict['bos_token'] = self.vocab[self.bos_token]
        if self.eos_token and self.eos_token in self.vocab:
            special_dict['eos_token'] = self.vocab[self.eos_token]
        if self.unk_token and self.unk_token in self.vocab:
            special_dict['unk_token'] = self.vocab[self.unk_token]
        if self.pad_token and self.pad_token in self.vocab:
            special_dict['pad_token'] = self.vocab[self.pad_token]
        
        return special_dict
    
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Apply chat template to messages with DeepSeek-R1 thinking support.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            add_generation_prompt: Whether to add generation prompt.
            
        Returns:
            Formatted chat text.
        """
        formatted_text = ""
        system_prompt = ""
        
        # Extract system prompt first
        for message in messages:
            if message.get('role') == 'system':
                system_prompt = message.get('content', '')
                break
        
        # Add BOS token and system prompt
        if self.bos_token:
            formatted_text += self.bos_token
        if system_prompt:
            formatted_text += system_prompt
        
        # Process messages
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                continue  # Already handled above
            elif role == 'user':
                formatted_text += f"<｜User｜>{content}"
            elif role == 'assistant':
                # Handle thinking content - extract only the final response
                if '</think>' in content:
                    content = content.split('</think>')[-1].strip()
                formatted_text += f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>"
        
        # Add generation prompt with thinking capability
        if add_generation_prompt:
            formatted_text += "<｜Assistant｜><think>\n"
        
        return formatted_text
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get BOS token ID."""
        if self.bos_token and self.bos_token in self.vocab:
            return self.vocab[self.bos_token]
        return None
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        if self.eos_token and self.eos_token in self.vocab:
            return self.vocab[self.eos_token]
        return None
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get PAD token ID."""
        if self.pad_token and self.pad_token in self.vocab:
            return self.vocab[self.pad_token]
        return None