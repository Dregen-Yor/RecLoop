# OneRec Single Model Evaluation Script Usage Guide

## Introduction

`eval_single_model.py` is a simplified evaluation script for evaluating OneRec models trained with `sft_simulation.py`.

## Features

- ✅ **Simplified process**: Direct evaluation without manual data splitting and merging
- ✅ **Auto-detection**: Automatically finds test files and info files
- ✅ **Complete evaluation**: Includes inference and metric calculation (NDCG, HR)
- ✅ **Flexible configuration**: Supports custom batch size, beam search, and other parameters

## Prerequisites

Ensure the following files exist:

1. **Model checkpoint**: Trained model directory, for example:
   ```
   simulation_storage/.../output/checkpoint-cycle-1/
   ```

2. **Test data**: Format `{data_name}_cycle_{cycle}_test.csv`, located in:
   ```
   {data_dir}/train/  or  {data_dir}/test/
   ```

3. **Info file**: Item information file, located in:
   ```
   recommenders/generativerec_onerec/data/{data_name}/info/{data_name}*.txt
   ```

## Usage

### Basic Usage

Evaluate the first cycle model (using default parameters):

```bash
cd recommenders/generativerec_onerec

python eval_single_model.py \
    --model_path /path/to/checkpoint-cycle-1 \
    --data_dir /path/to/onerec_data \
    --data_name Toys_and_Games
```

### Complete Example

Assuming your directory structure is as follows:

```
simulation_storage/
  simulation_storage_generative_{Toys_and_Games}_{OneRec}_noexclude_base_model_0.5B/
    onerec_data/
      train/
        Toys_and_Games_cycle_1_test.csv
        Toys_and_Games_cycle_1_train.csv
        Toys_and_Games_cycle_1_valid.csv
    output/
      checkpoint-cycle-1/
        adapter_config.json
        adapter_model.safetensors
        ...
```

Run evaluation:

```bash
python eval_single_model.py \
    --model_path simulation_storage/simulation_storage_generative_{Toys_and_Games}_{OneRec}_noexclude_base_model_0.5B/output/checkpoint-cycle-1 \
    --data_dir simulation_storage/simulation_storage_generative_{Toys_and_Games}_{OneRec}_noexclude_base_model_0.5B/onerec_data \
    --data_name Toys_and_Games \
    --cycle 1 \
    --gpu_id 0
```

### Full Parameter List

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Model checkpoint path (required) | - |
| `--data_dir` | Data directory (required) | - |
| `--data_name` | Dataset name (required) | - |
| `--cycle` | Evaluation cycle | 1 |
| `--gpu_id` | GPU ID to use | "0" |
| `--batch_size` | Batch size | 8 |
| `--num_beams` | Beam search count | 50 |
| `--max_new_tokens` | Maximum generated tokens | 256 |
| `--length_penalty` | Length penalty | 0.0 |
| `--seed` | Random seed | 42 |

### Advanced Usage

Evaluate the 5th cycle model with larger batch size and more beams:

```bash
python eval_single_model.py \
    --model_path .../checkpoint-cycle-5 \
    --data_dir .../onerec_data \
    --data_name Toys_and_Games \
    --cycle 5 \
    --batch_size 16 \
    --num_beams 100 \
    --gpu_id 1
```

## Output Results

The script will generate in `recommenders/generativerec_onerec/results/{model_name}/` directory:

1. **JSON result file**: `result_{data_name}_cycle_{cycle}.json`
   - Contains prediction results for each test sample

2. **Terminal output**: NDCG and HR metrics
   ```
   NDCG:  [0.xxx 0.xxx 0.xxx 0.xxx 0.xxx 0.xxx]  # @1, @3, @5, @10, @20, @50
   HR:    [0.xxx 0.xxx 0.xxx 0.xxx 0.xxx 0.xxx]  # @1, @3, @5, @10, @20, @50
   ```

## Supported Datasets

- `Toys_and_Games`
- `Office_Products`
- `Industrial_and_Scientific`
- `Sports`
- `Books`

## Common Issues

### Q1: Cannot find test file

**A**: Check the following:
- Is the `data_dir` path correct?
- Is the test file present in `train/` or `test/` subdirectory?
- Is the file named in the format `{data_name}_cycle_{cycle}_test.csv`?

### Q2: Cannot find info file

**A**: Ensure the following exists in the project root directory:
```
recommenders/generativerec_onerec/data/{data_name}/info/{data_name}*.txt
```

### Q3: GPU out of memory

**A**: Reduce `batch_size` or `num_beams`:
```bash
python eval_single_model.py \
    --batch_size 4 \
    --num_beams 20 \
    ...
```

### Q4: How to evaluate multiple cycles?

**A**: Run the script multiple times, modifying `--cycle` and `--model_path` parameters each time:
```bash
# Evaluate cycle 1
python eval_single_model.py --model_path .../checkpoint-cycle-1 --cycle 1 ...

# Evaluate cycle 2
python eval_single_model.py --model_path .../checkpoint-cycle-2 --cycle 2 ...
```

## Comparison with original evaluate.sh

| Feature | evaluate.sh | eval_single_model.py |
|---------|-------------|----------------------|
| Data splitting | ✅ Uses split.py | ❌ Not needed |
| Result merging | ✅ Uses merge.py | ❌ Not needed |
| Multi-GPU parallel | ✅ Supported | ❌ Single GPU |
| Usage complexity | Medium | Low |
| Use case | Production environment, large-scale evaluation | Quick evaluation, debugging |

## Technical Details

Evaluation process:
1. Validate model and data paths
2. Find test file (in `train/` or `test/` directory)
3. Find info file (in project root directory)
4. Call `evaluate.py` for inference generation
5. Call `calc.py` to calculate NDCG and HR metrics
6. Output results to terminal and JSON file

## Dependencies

- `fire`: Command-line parameter parsing
- `evaluate.py`: Inference generation
- `calc.py`: Metric calculation

## Contribution

To add new features or report issues, please contact the project maintainers.
