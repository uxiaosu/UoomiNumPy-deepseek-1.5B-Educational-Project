#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Weight Mapping Script

This script regenerates the weight_mapping.json file to include all weight files.
"""

import json
import os
from pathlib import Path

def generate_weight_mapping(weights_dir: str):
    """Generate complete weight mapping from .npy files.
    
    Args:
        weights_dir: Directory containing .npy weight files
    """
    weights_path = Path(weights_dir)
    
    if not weights_path.exists():
        print(f"❌ Weights directory not found: {weights_path}")
        return
    
    print(f"🔧 Generating weight mapping for: {weights_path}")
    
    mapping = {}
    
    # Process all .npy files
    npy_files = list(weights_path.glob("*.npy"))
    print(f"📁 Found {len(npy_files)} .npy files")
    
    for npy_file in npy_files:
        file_key = npy_file.stem  # filename without extension
        
        # Convert underscore-separated filename back to proper weight name
        # Handle different weight naming patterns
        if file_key == 'model_embed_tokens_weight':
            original_name = 'model.embed_tokens.weight'
        elif file_key == 'model_norm_weight':
            original_name = 'model.norm.weight'
        elif file_key == 'lm_head_weight':
            original_name = 'lm_head.weight'
        elif file_key.startswith('model_layers_'):
            # For layer weights: model_layers_0_self_attn_q_proj_weight -> model.layers.0.self_attn.q_proj.weight
            parts = file_key.split('_')
            if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers':
                layer_num = parts[2]
                component_parts = parts[3:]
                component_name = '_'.join(component_parts)
                
                # Convert remaining parts to proper dot notation
                if 'self_attn' in component_name:
                    # Convert self_attn_k_proj_weight to self_attn.k_proj.weight
                    component_name = component_name.replace('self_attn_', 'self_attn.')
                    component_name = component_name.replace('_proj_weight', '_proj.weight')
                    component_name = component_name.replace('_bias', '.bias')
                elif 'mlp' in component_name:
                    # Convert mlp_gate_proj_weight to mlp.gate_proj.weight
                    component_name = component_name.replace('mlp_', 'mlp.')
                    component_name = component_name.replace('_proj_weight', '_proj.weight')
                elif 'layernorm' in component_name:
                    # Convert input_layernorm_weight to input_layernorm.weight
                    component_name = component_name.replace('_weight', '.weight')
                
                original_name = f"model.layers.{layer_num}.{component_name}"
            else:
                original_name = file_key.replace('_', '.')
        else:
            original_name = file_key.replace('_', '.')
        
        mapping[file_key] = original_name
        print(f"   {file_key} -> {original_name}")
    
    # Save the mapping
    mapping_path = weights_path / "weight_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated weight mapping with {len(mapping)} entries")
    print(f"💾 Saved to: {mapping_path}")

def main():
    """Main function."""
    weights_dir = "examples/weights/deepseek_numpy_weights"
    
    print("🔧 Weight Mapping Fix Tool")
    print("=" * 50)
    
    generate_weight_mapping(weights_dir)
    
    print("\n✅ Weight mapping fix completed!")
    print("You can now try running the chat interface again.")

if __name__ == "__main__":
    main()