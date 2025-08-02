4#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Conversion Example for UoomiNumPy deepseek Educational Project

This example demonstrates how to convert model weights from different formats
(safetensors, PyTorch, etc.) to NumPy format for use with the pure NumPy implementation.
"""

import sys
from pathlib import Path
import os
import time

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from weight_loader import WeightLoader
from config import ModelConfig


def convert_model_weights(source_dir: str, target_dir: str, format_type: str = "auto"):
    """Convert model weights to NumPy format.
    
    Args:
        source_dir: Directory containing the original model
        target_dir: Directory to save converted weights
        format_type: Source format ('safetensors', 'pytorch', 'auto')
    """
    print(f"🔄 Converting weights from {source_dir} to {target_dir}")
    print(f"📁 Source format: {format_type}")
    print("=" * 60)
    
    # Initialize weight loader
    loader = WeightLoader(source_dir)
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Load configuration
        print("📋 Loading model configuration...")
        config_path = os.path.join(source_dir, "config.json")
        if os.path.exists(config_path):
            config = ModelConfig.from_json(config_path)
            print(f"✅ Configuration loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        else:
            print("⚠️  No config.json found, using default configuration")
            config = ModelConfig()
        
        # Save configuration to target directory
        config.to_json(os.path.join(target_dir, "config.json"))
        
        # Convert weights based on format
        start_time = time.time()
        
        # Load weights using the unified method
        print("🔧 Loading weights...")
        weights = loader.load_weights()
        
        # Save weights in NumPy format
        print("💾 Saving weights in NumPy format...")
        loader.save_weights_numpy(weights, target_dir)
        
        # Optimize weights (optional)
        print("⚡ Optimizing weights...")
        optimized_weights = loader.optimize_weights(weights)
        
        conversion_time = time.time() - start_time
        
        # Validate conversion
        print("✅ Validating conversion...")
        is_valid = loader.validate_weights(weights)
        
        if is_valid:
            print("✅ Weight conversion completed successfully!")
            print(f"⏱️  Conversion time: {conversion_time:.2f} seconds")
            print(f"📊 Total parameters: {sum(w.size for w in weights.values()):,}")
            total_size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)
            print(f"💾 Total size: {total_size_mb:.2f} MB")
        else:
            print("❌ Weight validation failed!")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


def analyze_weights(weights_dir: str):
    """Analyze converted weights.
    
    Args:
        weights_dir: Directory containing NumPy weights
    """
    print(f"🔍 Analyzing weights in {weights_dir}")
    print("=" * 50)
    
    loader = WeightLoader(weights_dir)
    
    try:
        # Load weights
        weights = loader.load_weights_numpy(weights_dir)
        
        # Analyze structure
        print("📊 Weight Structure Analysis:")
        print("-" * 30)
        
        total_params = 0
        layer_info = {}
        
        for name, weight in weights.items():
            params = weight.size
            total_params += params
            
            # Categorize by layer type
            if "embed" in name:
                category = "Embedding"
            elif "norm" in name:
                category = "Normalization"
            elif "attn" in name:
                category = "Attention"
            elif "mlp" in name:
                category = "MLP"
            elif "lm_head" in name:
                category = "Output"
            else:
                category = "Other"
            
            if category not in layer_info:
                layer_info[category] = {"count": 0, "params": 0, "weights": []}
            
            layer_info[category]["count"] += 1
            layer_info[category]["params"] += params
            layer_info[category]["weights"].append((name, weight.shape, params))
        
        # Display analysis
        print(f"Total Parameters: {total_params:,}")
        print(f"Total Weights: {len(weights)}")
        print()
        
        for category, info in layer_info.items():
            percentage = (info["params"] / total_params) * 100
            print(f"{category}:")
            print(f"  Count: {info['count']} weights")
            print(f"  Parameters: {info['params']:,} ({percentage:.1f}%)")
            
            # Show top 3 largest weights in category
            sorted_weights = sorted(info["weights"], key=lambda x: x[2], reverse=True)
            for name, shape, params in sorted_weights[:3]:
                print(f"    {name}: {shape} ({params:,} params)")
            print()
        
        # Memory usage analysis
        print("💾 Memory Usage Analysis:")
        print("-" * 30)
        
        memory_usage = {}
        for name, weight in weights.items():
            dtype_size = weight.dtype.itemsize
            memory_mb = (weight.size * dtype_size) / (1024 * 1024)
            memory_usage[name] = memory_mb
        
        total_memory = sum(memory_usage.values())
        print(f"Total Memory: {total_memory:.2f} MB")
        
        # Show top 5 memory consumers
        sorted_memory = sorted(memory_usage.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 Memory Consumers:")
        for name, memory in sorted_memory[:5]:
            percentage = (memory / total_memory) * 100
            print(f"  {name}: {memory:.2f} MB ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error analyzing weights: {e}")


def compare_weights(original_dir: str, converted_dir: str):
    """Compare original and converted weights.
    
    Args:
        original_dir: Directory with original weights
        converted_dir: Directory with converted NumPy weights
    """
    print(f"⚖️  Comparing weights: {original_dir} vs {converted_dir}")
    print("=" * 60)
    
    loader = WeightLoader(original_dir)
    
    try:
        # Load both sets of weights
        print("📥 Loading original weights...")
        original_loader = WeightLoader(original_dir)
        original_weights = original_loader.load_weights()
        
        print("📥 Loading converted weights...")
        converted_weights = loader.load_weights_numpy(converted_dir)
        
        # Compare
        print("🔍 Comparing weights...")
        
        # Check if all weights are present
        original_keys = set(original_weights.keys())
        converted_keys = set(converted_weights.keys())
        
        missing_in_converted = original_keys - converted_keys
        extra_in_converted = converted_keys - original_keys
        
        if missing_in_converted:
            print(f"⚠️  Missing in converted: {missing_in_converted}")
        
        if extra_in_converted:
            print(f"ℹ️  Extra in converted: {extra_in_converted}")
        
        # Compare common weights
        common_keys = original_keys & converted_keys
        differences = []
        
        for key in common_keys:
            orig_weight = original_weights[key]
            conv_weight = converted_weights[key]
            
            # Convert original to numpy if needed
            if hasattr(orig_weight, 'numpy'):
                orig_weight = orig_weight.numpy()
            elif hasattr(orig_weight, 'detach'):
                orig_weight = orig_weight.detach().cpu().numpy()
            
            # Compare shapes
            if orig_weight.shape != conv_weight.shape:
                differences.append(f"{key}: shape mismatch {orig_weight.shape} vs {conv_weight.shape}")
                continue
            
            # Compare values (with tolerance for floating point)
            import numpy as np
            if not np.allclose(orig_weight, conv_weight, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(orig_weight - conv_weight))
                differences.append(f"{key}: value mismatch (max diff: {max_diff:.2e})")
        
        # Report results
        if not differences:
            print("✅ All weights match perfectly!")
        else:
            print(f"⚠️  Found {len(differences)} differences:")
            for diff in differences[:10]:  # Show first 10
                print(f"   - {diff}")
            if len(differences) > 10:
                print(f"   ... and {len(differences) - 10} more")
        
        print(f"\n📊 Comparison Summary:")
        print(f"  Original weights: {len(original_keys)}")
        print(f"  Converted weights: {len(converted_keys)}")
        print(f"  Common weights: {len(common_keys)}")
        print(f"  Differences found: {len(differences)}")
        
    except Exception as e:
        print(f"❌ Error comparing weights: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for weight conversion examples."""
    print("⚙️  UoomiNumPy deepseek Model - Weight Conversion Examples")
    print("=" * 70)
    
    # Example paths (update these to your actual paths)
    source_model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    target_weights_dir = "weights/deepseek_numpy_weights"
    
    print("Choose an option:")
    print("1. Convert weights to NumPy format")
    print("2. Analyze existing NumPy weights")
    print("3. Compare original and converted weights")
    print("4. Full conversion pipeline")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Convert weights
            print(f"\nSource directory: {source_model_dir}")
            print(f"Target directory: {target_weights_dir}")
            
            format_choice = input("\nWeight format (safetensors/pytorch/auto): ").strip().lower()
            if not format_choice:
                format_choice = "auto"
            
            convert_model_weights(source_model_dir, target_weights_dir, format_choice)
            
        elif choice == "2":
            # Analyze weights
            weights_dir = input(f"\nWeights directory [{target_weights_dir}]: ").strip()
            if not weights_dir:
                weights_dir = target_weights_dir
            
            analyze_weights(weights_dir)
            
        elif choice == "3":
            # Compare weights
            original_dir = input(f"\nOriginal model directory [{source_model_dir}]: ").strip()
            if not original_dir:
                original_dir = source_model_dir
            
            converted_dir = input(f"\nConverted weights directory [{target_weights_dir}]: ").strip()
            if not converted_dir:
                converted_dir = target_weights_dir
            
            compare_weights(original_dir, converted_dir)
            
        elif choice == "4":
            # Full pipeline
            print("\n🚀 Running full conversion pipeline...")
            
            # Step 1: Convert
            convert_model_weights(source_model_dir, target_weights_dir, "auto")
            
            # Step 2: Analyze
            print("\n" + "=" * 50)
            analyze_weights(target_weights_dir)
            
            # Step 3: Compare
            print("\n" + "=" * 50)
            compare_weights(source_model_dir, target_weights_dir)
            
            print("\n✅ Full pipeline completed!")
            
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()