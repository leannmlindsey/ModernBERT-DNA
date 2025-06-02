# DNA Sequence Classification Fine-tuning

This directory contains the setup for fine-tuning ModernBERT-DNA on three major DNA sequence classification benchmarks:
- **NT v2** (Nucleotide Transformer v2): 18 tasks
- **GB** (Genomic Benchmarks): 9 tasks  
- **GUE** (Genome Understanding Evaluation): 28+ tasks

## Benchmarks Overview

### 1. Nucleotide Transformer v2 (NT v2)
The NT v2 benchmark consists of 18 DNA sequence classification tasks across four categories:
- **Histone Modifications** (10 tasks): H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K9me3, H4K20me1
- **Enhancers** (2 tasks): enhancers, enhancers_types
- **Promoters** (3 tasks): promoter_all, promoter_no_tata, promoter_tata
- **Splice Sites** (3 tasks): splice_sites_acceptors, splice_sites_all, splice_sites_donors

### 2. Genomic Benchmarks (GB)
The GB collection contains 9 curated genomic sequence classification datasets:
- **Human tasks**: human_nontata_promoters, human_ocr_ensembl, human_enhancers_ensembl, human_enhancers_cohn, human_ensembl_regulatory
- **Mouse tasks**: mouse_enhancers_ensembl
- **Other organisms**: drosophila_enhancers_stark, demo_human_or_worm, demo_coding_vs_intergenomic

### 3. Genome Understanding Evaluation (GUE)
GUE is a comprehensive multi-species benchmark with 28+ datasets across 7 task categories:
- **Promoter prediction**: human/mouse promoter and core promoter detection
- **Splice site prediction**: acceptor/donor sites for human and mouse
- **COVID variant classification**: SARS-CoV-2 variant identification
- **Epigenetic marks**: H3K4me3, H3K27me3, H3K36me3, H3K9me3, H3K27ac
- **TF binding sites**: CTCF, NFκB, CEBPB, FOXA1
- **Species-specific tasks**: yeast UTRs, fungi promoters, virus classification
- **GUE+ (long-range)**: Tasks with sequences up to 10,000bp

## Prerequisites

1. **Data**: Download datasets and organize as follows:
   ```
   ../DATA/
   └── <task_name>/
       ├── train.csv  # Standardized format with "sequence" and "label" columns
       ├── dev.csv    # (or test.csv for GB/GUE tasks)
       └── test.csv
   ```
   
   **Standardized CSV Format**: All tasks can use CSV files with two columns:
   - `sequence`: DNA sequence
   - `label`: Class label (integer or string)
   
   **Alternative formats** (if CSV not available):
   - GB tasks: TSV files with "sequence" and "label" columns
   - GUE tasks: JSON files with "sequence" and "label" fields, or TSV format

2. **Pretrained Checkpoints**: Update the checkpoint paths in the YAML configs:
   - For BPE model: Update `pretrained_checkpoint` in respective base YAML files
   - For character model: Update `pretrained_checkpoint` in `modernbert_dna_char.yaml`

## Usage

### Fine-tune Individual Tasks

#### NT v2 Tasks
```bash
# Fine-tune with BPE tokenization (default)
./run_dna_finetuning.sh H3K27ac bpe

# Fine-tune with character-level tokenization
./run_dna_finetuning.sh H3K27ac char
```

#### GB Tasks
```bash
# Fine-tune a GB task
./run_gb_finetuning.sh human_nontata_promoters bpe

# With additional arguments
./run_gb_finetuning.sh human_enhancers_ensembl bpe --lr 5e-5
```

#### GUE Tasks
```bash
# Fine-tune a GUE task
./run_gue_finetuning.sh human_promoter bpe

# Fine-tune a long-range GUE+ task
./run_gue_finetuning.sh human_chromatin_interactions bpe
```

### Fine-tune All Tasks in a Benchmark

```bash
# Run all NT v2 tasks
./run_all_dna_tasks.sh bpe

# Run all GB tasks
./run_all_gb_tasks.sh bpe

# Run all GUE tasks
./run_all_gue_tasks.sh bpe all

# Run only standard GUE tasks (excluding long-range)
./run_all_gue_tasks.sh bpe standard

# Run only GUE+ long-range tasks
./run_all_gue_tasks.sh bpe long
```

## Configuration Files

### Main Scripts
- `dna_sequence_classification.py`: Main training script supporting all three benchmarks
- Shell scripts for each benchmark:
  - NT v2: `run_dna_finetuning.sh`, `run_all_dna_tasks.sh`
  - GB: `run_gb_finetuning.sh`, `run_all_gb_tasks.sh`
  - GUE: `run_gue_finetuning.sh`, `run_all_gue_tasks.sh`

### YAML Configurations
```
yamls/dna_finetuning/
├── modernbert_dna_base.yaml      # Base config for NT v2 with BPE
├── modernbert_dna_char.yaml      # Base config for character tokenization
├── histone_modifications.yaml    # NT v2 histone tasks
├── enhancers.yaml               # NT v2 enhancer tasks
├── promoters.yaml               # NT v2 promoter tasks
├── splice_sites.yaml            # NT v2 splice site tasks
├── gb/
│   ├── gb_base.yaml            # Base config for GB tasks
│   └── gb_tasks.yaml           # Task-specific GB settings
└── gue/
    ├── gue_base.yaml           # Base config for GUE tasks
    ├── gue_tasks.yaml          # Standard GUE task settings
    └── gue_long.yaml           # GUE+ long-range task settings
```

## Key Features

1. **Multi-format Support**: Handles CSV (NT v2), TSV (GB), and JSON/TSV (GUE) data formats
2. **Dynamic Sequence Length**: Each task uses appropriate max sequence lengths
3. **Two Tokenization Options**: 
   - BPE tokenization using DNABERT-2 tokenizer
   - Character-level tokenization with custom DNA tokenizer
4. **Efficient Training**: 
   - Flash Attention 2/3 support
   - Mixed precision training
   - Gradient accumulation for long sequences

## Output Structure

```
outputs/
├── dna_finetuning/          # NT v2 results
│   └── <task_name>_<model_type>/
├── gb/                       # GB results  
│   └── <task_name>_<model_type>/
└── gue/                      # GUE results
    └── <task_name>_<model_type>/

logs/
├── dna_finetuning_<model_type>_<timestamp>/
├── gb_finetuning_<model_type>_<timestamp>/
└── gue_finetuning_<model_type>_<timestamp>/
```

## Hyperparameters

Default hyperparameters (can be overridden via CLI):
- Learning rate: 3e-5
- Batch size: 64 (train), 128 (eval)
- Epochs: 10
- Warmup: 50 batches (NT v2), 6% of training (GB/GUE)
- Optimizer: AdamW with weight decay 0.01
- Mixed precision: FP16

For GUE+ long-range tasks:
- Reduced batch size: 32 (train), 64 (eval)
- Larger sliding window: 256
- More frequent global attention

## Monitoring

Training progress can be monitored through:
- Console output with progress bars
- Log files in the `logs/` directory
- Optional WandB integration (uncomment in YAML configs)

## Troubleshooting

1. **Out of Memory**: 
   - Reduce `global_train_batch_size` or `device_train_microbatch_size`
   - For long sequences, use gradient accumulation

2. **Slow Training**: 
   - Ensure Flash Attention 2 is installed and enabled
   - Check that you're using mixed precision training

3. **Data Not Found**: 
   - Verify data paths match expected structure
   - GB uses `.tsv` files, GUE can use `.json` or `.tsv`
   - NT v2 uses `.csv` files with `dev.csv` for validation

4. **Data Formats**:
   - **Preferred**: CSV with "sequence" and "label" columns (works for all benchmarks)
   - **Alternatives** (auto-detected if CSV not found):
     - NT v2: CSV with positional columns (sequence in first column, label in second)
     - GB: TSV with "sequence" and "label" columns
     - GUE: JSON with "sequence" and "label" fields, or TSV format