#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Loader - Pure NumPy Implementation

Handles loading and converting model weights from various formats.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import struct
import os

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


class WeightLoader:
    """Handles loading model weights from different formats."""
    
    def __init__(self, model_dir: str):
        """Initialize weight loader.
        
        Args:
            model_dir: Directory containing model files.
        """
        self.model_dir = Path(model_dir)
        self.weights_cache = {}
    
    def load_weights(self) -> Dict[str, np.ndarray]:
        """Load all model weights.
        
        Returns:
            Dictionary mapping weight names to numpy arrays.
        """
        weights = {}
        
        # Try to load from safetensors first
        safetensors_path = self.model_dir / "model.safetensors"
        if safetensors_path.exists() and SAFETENSORS_AVAILABLE:
            print(f"📦 Loading weights from safetensors: {safetensors_path}")
            weights.update(self._load_safetensors(safetensors_path))
        
        # Try to load from PyTorch files
        pytorch_files = list(self.model_dir.glob("pytorch_model*.bin"))
        if pytorch_files:
            print(f"📦 Loading weights from PyTorch files: {len(pytorch_files)} files")
            for pytorch_file in pytorch_files:
                weights.update(self._load_pytorch_weights(pytorch_file))
        
        # Try to load from numpy files
        numpy_files = list(self.model_dir.glob("*.npy"))
        if numpy_files:
            print(f"📦 Loading weights from numpy files: {len(numpy_files)} files")
            weights.update(self.load_weights_numpy(str(self.model_dir)))
            return weights  # Return early since we loaded all weights
        
        if not weights:
            raise FileNotFoundError(f"No compatible weight files found in {self.model_dir}")
        
        print(f"✅ Loaded {len(weights)} weight tensors")
        print(f"🔍 Debug: First 10 weight keys: {list(weights.keys())[:10]}")
        return weights
    
    def _load_safetensors(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load weights from safetensors file."""
        weights = {}
        
        try:
            # Try PyTorch framework first to handle bfloat16
            import torch
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert to numpy and handle bfloat16
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()  # Convert to float32
                    weights[key] = tensor.numpy()
                    print(f"   Loaded: {key} -> {weights[key].shape} {weights[key].dtype}")
        except ImportError:
            # Fallback to numpy framework
            with safe_open(file_path, framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    weights[key] = tensor
                    print(f"   Loaded: {key} -> {tensor.shape} {tensor.dtype}")
        
        return weights
    
    def _load_pytorch_weights(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load weights from PyTorch file."""
        try:
            import torch
            
            # Load PyTorch state dict
            state_dict = torch.load(file_path, map_location='cpu')
            
            weights = {}
            for key, tensor in state_dict.items():
                # Convert to numpy
                if hasattr(tensor, 'numpy'):
                    weights[key] = tensor.numpy()
                else:
                    weights[key] = np.array(tensor)
                
                print(f"   Loaded: {key} -> {weights[key].shape} {weights[key].dtype}")
            
            return weights
            
        except ImportError:
            print("Warning: PyTorch not available. Cannot load .bin files.")
            return {}
    
    def _load_numpy_weights(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load weights from numpy file."""
        weights = {}
        
        # Extract weight name from filename
        weight_name = file_path.stem
        
        # Load numpy array
        tensor = np.load(file_path)
        weights[weight_name] = tensor
        
        print(f"   Loaded: {weight_name} -> {tensor.shape} {tensor.dtype}")
        
        return weights
    
    def save_weights_numpy(self, weights: Dict[str, np.ndarray], output_dir: str):
        """Save weights in numpy format.
        
        Args:
            weights: Dictionary of weight tensors.
            output_dir: Output directory for saved weights.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving weights to numpy format: {output_path}")
        
        for name, tensor in weights.items():
            # Clean up name for filename
            clean_name = name.replace('.', '_').replace('/', '_')
            file_path = output_path / f"{clean_name}.npy"
            
            np.save(file_path, tensor)
            print(f"   Saved: {name} -> {file_path}")
        
        # Save weight mapping
        mapping = {clean_name.replace('.', '_').replace('/', '_'): name for name in weights.keys()}
        mapping_path = output_path / "weight_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✅ Saved {len(weights)} weight files")
    
    def load_weights_numpy(self, weights_dir: str) -> Dict[str, np.ndarray]:
        """Load weights from numpy format.
        
        Args:
            weights_dir: Directory containing numpy weight files.
            
        Returns:
            Dictionary of weight tensors.
        """
        weights_path = Path(weights_dir)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_path}")
        
        print(f"📂 Loading weights from numpy format: {weights_path}")
        
        weights = {}
        
        # Load weight mapping if available
        mapping_path = weights_path / "weight_mapping.json"
        name_mapping = {}
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                name_mapping = json.load(f)
        
        # Load all .npy files
        for npy_file in weights_path.glob("*.npy"):
            if npy_file.name == "weight_mapping.json":
                continue
            
            file_key = npy_file.stem
            original_name = name_mapping.get(file_key, file_key.replace('_', '.'))
            
            tensor = np.load(npy_file)
            weights[original_name] = tensor
            
            print(f"   Loaded: {file_key} -> {original_name} -> {tensor.shape} {tensor.dtype}")
        
        print(f"✅ Loaded {len(weights)} weight tensors")
        return weights
    
    def convert_weights_format(
        self, 
        input_format: str, 
        output_format: str, 
        output_dir: str
    ):
        """Convert weights between different formats.
        
        Args:
            input_format: Input format ('safetensors', 'pytorch', 'numpy').
            output_format: Output format ('numpy', 'json').
            output_dir: Output directory.
        """
        print(f"🔄 Converting weights from {input_format} to {output_format}")
        
        # Load weights
        weights = self.load_weights()
        
        # Convert and save
        if output_format == 'numpy':
            self.save_weights_numpy(weights, output_dir)
        elif output_format == 'json':
            self._save_weights_json(weights, output_dir)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_weights_json(self, weights: Dict[str, np.ndarray], output_dir: str):
        """Save weights metadata in JSON format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {}
        for name, tensor in weights.items():
            metadata[name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'size': int(tensor.size),
                'memory_mb': float(tensor.nbytes / (1024 * 1024))
            }
        
        json_path = output_path / "weights_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved weights metadata: {json_path}")
    
    def get_weight_info(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get information about loaded weights.
        
        Args:
            weights: Dictionary of weight tensors.
            
        Returns:
            Dictionary containing weight statistics.
        """
        total_params = sum(tensor.size for tensor in weights.values())
        total_memory = sum(tensor.nbytes for tensor in weights.values())
        
        info = {
            'total_tensors': len(weights),
            'total_parameters': total_params,
            'total_memory_mb': total_memory / (1024 * 1024),
            'total_memory_gb': total_memory / (1024 * 1024 * 1024),
            'tensors': {}
        }
        
        for name, tensor in weights.items():
            info['tensors'][name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'parameters': int(tensor.size),
                'memory_mb': float(tensor.nbytes / (1024 * 1024))
            }
        
        return info
    
    def validate_weights(self, weights: Dict[str, np.ndarray]) -> bool:
        """Validate loaded weights.
        
        Args:
            weights: Dictionary of weight tensors.
            
        Returns:
            True if weights are valid, False otherwise.
        """
        print("🔍 Validating weights...")
        
        required_weights = [
            'model.embed_tokens.weight',
            'model.norm.weight',
            'lm_head.weight'
        ]
        
        # Check for required weights
        missing_weights = []
        for weight_name in required_weights:
            if weight_name not in weights:
                missing_weights.append(weight_name)
        
        if missing_weights:
            print(f"❌ Missing required weights: {missing_weights}")
            return False
        
        # Check for NaN or infinite values
        for name, tensor in weights.items():
            if np.isnan(tensor).any():
                print(f"❌ NaN values found in {name}")
                return False
            
            if np.isinf(tensor).any():
                print(f"❌ Infinite values found in {name}")
                return False
        
        print("✅ All weights are valid")
        return True
    
    def optimize_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimize weights for inference.
        
        Args:
            weights: Dictionary of weight tensors.
            
        Returns:
            Optimized weight tensors.
        """
        print("⚡ Optimizing weights for inference...")
        
        optimized_weights = {}
        
        for name, tensor in weights.items():
            # Convert to float32 for better performance
            if tensor.dtype != np.float32:
                optimized_tensor = tensor.astype(np.float32)
                print(f"   Converted {name}: {tensor.dtype} -> {optimized_tensor.dtype}")
            else:
                optimized_tensor = tensor
            
            # Ensure contiguous memory layout
            if not optimized_tensor.flags['C_CONTIGUOUS']:
                optimized_tensor = np.ascontiguousarray(optimized_tensor)
                print(f"   Made contiguous: {name}")
            
            optimized_weights[name] = optimized_tensor
        
        print("✅ Weight optimization completed")
        return optimized_weights