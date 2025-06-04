# Future Development

This directory contains code that may be adapted for ModernBERT-DNA in the future but is not currently in use.

## Files

### `wandb_log_live_eval.py`
- **Purpose**: Aggregates evaluation metrics from multiple WandB runs and creates consolidated views
- **Current State**: Designed for original ModernBERT's GLUE/SuperGLUE tasks
- **Future Work**: Could be modified to aggregate DNA benchmark results (NTv2, GUE, GB) across multiple seeds and create unified performance tracking
- **Key Features**:
  - Monitors completed runs periodically
  - Averages metrics across seeds
  - Creates time-series view of model performance
  - Does NOT submit new runs, only aggregates existing results

To adapt for DNA tasks, would need:
- Update task names and metric mappings for DNA benchmarks
- Modify run name parsing pattern
- Update metric names to match DNA evaluation outputs