#!/usr/bin/env python3
"""
Memory-efficient script to convert DNA sequence data directly to MDS format
without using the Hugging Face datasets library intermediary.
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Iterator, Any
import gc

from streaming import MDSWriter
from tqdm import tqdm


def yield_dna_sequences(file_path: str, batch_size: int = 10000) -> Iterator[Dict[str, bytes]]:
    """Yield DNA sequences from a file in memory-efficient batches."""
    with open(file_path, 'r') as f:
        count = 0
        for line in f:
            if line.strip():
                # Convert to bytes as expected by MDSWriter
                yield {"text": line.strip().encode('utf-8')}
                count += 1
                
                # Free up memory periodically
                if count % batch_size == 0:
                    gc.collect()


def convert_to_mds(input_file: str, output_dir: str, compression: str = None, desc: str = "Converting") -> int:
    """Convert a DNA sequence file directly to MDS format in a memory-efficient way."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Count lines for progress reporting
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Define columns for MDSWriter
    columns = {"text": "str"}
    
    # Convert sequences to MDS format
    count = 0
    with MDSWriter(columns=columns, out=output_dir, compression=compression) as out:
        for sample in tqdm(yield_dna_sequences(input_file), desc=desc, total=total_lines):
            out.write(sample)
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert DNA sequence files directly to MDS format")
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Path to training DNA sequences (one per line)")
    parser.add_argument("--val_file", type=str, required=True, 
                        help="Path to validation DNA sequences (one per line)")
    parser.add_argument("--out_root", type=str, required=True, 
                        help="Root directory to save MDS-formatted files")
    parser.add_argument("--compression", type=str, default=None,
                        help="Compression format for MDS files")
    
    args = parser.parse_args()
    
    # Convert training data
    print("Converting training data...")
    train_count = convert_to_mds(
        args.train_file, 
        os.path.join(args.out_root, "train"),
        args.compression,
        "Converting train data"
    )
    
    # Convert validation data
    print("Converting validation data...")
    val_count = convert_to_mds(
        args.val_file, 
        os.path.join(args.out_root, "val"),
        args.compression,
        "Converting val data"
    )
    
    # Create metadata file
    metadata = {
        "dataset_name": "dna_sequences",
        "splits": {
            "train": {"num_sequences": train_count},
            "val": {"num_sequences": val_count}
        }
    }
    
    with open(os.path.join(args.out_root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Conversion complete!")
    print(f"Training sequences: {train_count}")
    print(f"Validation sequences: {val_count}")
    print(f"Data saved to: {args.out_root}")


if __name__ == "__main__":
    main()
