# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading DNA model checkpoints."""

import os
import torch
import tempfile


def extract_state_dict_from_composer_checkpoint(checkpoint_path: str) -> str:
    """Extract state dict from Composer checkpoint and save to temporary file.
    
    Args:
        checkpoint_path: Path to Composer checkpoint
        
    Returns:
        Path to temporary file containing just the state dict
    """
    # Load the full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract the model state dict
    if isinstance(checkpoint, dict) and 'state' in checkpoint:
        # This is a Composer checkpoint
        state_dict = checkpoint['state']['model']
    else:
        # This is already a state dict, just return the original path
        return checkpoint_path
    
    # Save the state dict to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save(state_dict, temp_file.name)
    temp_file.close()
    
    return temp_file.name


def load_pretrained_dna_model(model_class, model_args, checkpoint_path):
    """Load a pretrained DNA model, handling Composer checkpoint format.
    
    Args:
        model_class: The model class (e.g., flex_bert_module)
        model_args: Arguments to pass to model creation
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Loaded model
    """
    # First try to load directly
    try:
        # Extract state dict if needed
        state_dict_path = extract_state_dict_from_composer_checkpoint(checkpoint_path)
        
        # Update model args to use the extracted state dict
        model_args_copy = model_args.copy()
        model_args_copy['pretrained_checkpoint'] = state_dict_path
        
        # Create the model
        if hasattr(model_class, 'create_flex_bert_classification'):
            model = model_class.create_flex_bert_classification(**model_args_copy)
        elif hasattr(model_class, 'create_mosaic_bert_classification'):
            model = model_class.create_mosaic_bert_classification(**model_args_copy)
        elif hasattr(model_class, 'create_hf_bert_classification'):
            model = model_class.create_hf_bert_classification(**model_args_copy)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Clean up temp file if created
        if state_dict_path != checkpoint_path:
            os.unlink(state_dict_path)
            
        return model
        
    except Exception as e:
        # Clean up temp file on error
        if 'state_dict_path' in locals() and state_dict_path != checkpoint_path:
            try:
                os.unlink(state_dict_path)
            except:
                pass
        raise e