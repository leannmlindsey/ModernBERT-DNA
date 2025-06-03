#!/usr/bin/env python
"""Quick script to check what's in a checkpoint file."""

import torch
import sys

if len(sys.argv) != 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check what's in the checkpoint
if isinstance(checkpoint, dict):
    print("\nCheckpoint is a dictionary with keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # If it has 'state_dict' key, examine it
    if 'state_dict' in checkpoint:
        print("\nExamining 'state_dict':")
        state_dict = checkpoint['state_dict']
        
        # Get all unique prefixes
        prefixes = set()
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) > 1:
                prefixes.add(parts[0])
        
        print(f"Unique prefixes in state_dict: {prefixes}")
        
        # Show first 10 keys
        print("\nFirst 10 keys in state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}")
        
        # Check for model wrapper prefixes
        example_key = list(state_dict.keys())[0]
        print(f"\nExample key: {example_key}")
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if 'encoder.layers' in k]
        layer_numbers = set()
        for k in layer_keys:
            if 'encoder.layers.' in k:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts):
                        try:
                            layer_num = int(parts[i+1])
                            layer_numbers.add(layer_num)
                        except:
                            pass
        
        if layer_numbers:
            print(f"\nFound layers: {sorted(layer_numbers)}")
            print(f"Total number of layers: {len(layer_numbers)}")
    
    # Also check 'state' if it exists
    if 'state' in checkpoint:
        print("\nCheckpoint has 'state' key")
        if isinstance(checkpoint['state'], dict) and 'model' in checkpoint['state']:
            print("Found model in checkpoint['state']")
else:
    print("\nCheckpoint is not a dictionary, it's a:", type(checkpoint))
    
print("\nDone!")