"""
SFT Training Wrapper for OneRec Simulation
Simplified training interface for multi-round simulation
"""

import os
import sys
import fire
import glob
from sft import train  # Import the original training function


def train_simulation(
    # Simulation-specific params
    data_dir: str = "",
    output_dir: str = "",
    cycle: int = 1,
    data_name: str = "Toys_and_Games",
    
    # Model params
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prev_checkpoint: str = "",
    
    # Index and metadata
    sid_index_path: str = "",
    item_meta_path: str = "",
    
    # Training hyperparams (aligned with sft.py defaults)
    batch_size: int = 1024,
    micro_batch_size: int = 16,
    num_train_epochs: int = 10,
    learning_rate: float = 3e-4,
    freeze_LLM: bool = False,
    
    # Seed
    seed: int = 42,
    
    # Wandb (disabled for simulation)
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """
    SFT training wrapper - for simulation system
    
    Args:
        data_dir: Data directory (contains train/ subdirectory)
        output_dir: Output directory
        cycle: Current cycle
        data_name: Dataset name
        base_model: Base model path
        prev_checkpoint: Previous cycle's checkpoint path (optional)
        sid_index_path: SID index file path
        item_meta_path: Item metadata file path
        batch_size: Batch size
        micro_batch_size: Micro batch size
        num_train_epochs: Training epochs
        learning_rate: Learning rate
        freeze_LLM: Whether to freeze LLM parameters
        seed: Random seed
        wandb_project: Wandb project name (leave empty to disable)
        wandb_run_name: Wandb run name
    """
    
    print("=" * 60)
    print("OneRec SFT training wrapper - simulation mode")
    print("=" * 60)
    print(f"Dataset: {data_name}")
    print(f"Cycle: {cycle}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    train_dir = os.path.join(data_dir, "train")
    
    train_files = glob.glob(os.path.join(train_dir, f"{data_name}_cycle_{cycle}_train.csv"))
    if not train_files:
        train_files = glob.glob(os.path.join(train_dir, f"{data_name}_cycle_*_train.csv"))
    
    if not train_files:
        raise FileNotFoundError(f"Training file not found in {train_dir} (expected: {data_name}_cycle_{cycle}_train.csv)")
    
    train_file = sorted(train_files)[-1]
    print(f"Training file: {train_file}")
    
    valid_files = glob.glob(os.path.join(train_dir, f"{data_name}_cycle_{cycle}_valid.csv"))
    if not valid_files:
        valid_files = glob.glob(os.path.join(train_dir, f"{data_name}_cycle_*_valid.csv"))
    
    if not valid_files:
        print(f"⚠ Validation file not found, skipping validation")
        valid_file = ""
    else:
        valid_file = sorted(valid_files)[-1]
        print(f"Validation file: {valid_file}")
    
    if cycle > 1 and prev_checkpoint and os.path.exists(prev_checkpoint):
        model_to_use = prev_checkpoint
        train_from_scratch = False
        print(f"Continuing training from checkpoint: {prev_checkpoint}")
    else:
        model_to_use = base_model
        train_from_scratch = False if cycle > 1 else False
        print(f"Starting from base model: {base_model}")
    
    cycle_output_dir = os.path.join(output_dir, f"checkpoint-cycle-{cycle}")
    os.makedirs(cycle_output_dir, exist_ok=True)
    print(f"Checkpoint output directory: {cycle_output_dir}")
    
    info_dir = os.path.dirname(os.path.dirname(data_dir))
    onerec_base = os.path.join(info_dir, "recommenders", "generativerec_onerec", "data", data_name)
    if not os.path.exists(onerec_base):
        onerec_base = os.path.join("recommenders", "generativerec_onerec", "data", data_name)
    
    info_files = glob.glob(os.path.join(onerec_base, "info", f"{data_name}*.txt"))
    if not info_files:
        raise FileNotFoundError(f"Info file not found in {onerec_base}/info/")
    info_file = info_files[0]
    print(f"Info file: {info_file}")
    
    category_map = {
        "Toys_and_Games": "Toys_and_Games",
        "Office_Products": "Office_Products",
        "Industrial_and_Scientific": "Industrial_and_Scientific",
        "Sports": "Sports",
        "Books": "Books",
    }
    category = category_map.get(data_name, data_name)
    
    print(f"\nStarting training...")
    print(f"  batch_size: {batch_size}")
    print(f"  micro_batch_size: {micro_batch_size}")
    print(f"  num_train_epochs: {num_train_epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  freeze_LLM: {freeze_LLM}")
    print("=" * 60)
    
    try:
        train(
            # Model/data params
            base_model=model_to_use,
            train_file=train_file,
            eval_file=valid_file,
            output_dir=cycle_output_dir,
            sample=-1,
            seed=seed,
            
            # Training hyperparams
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            num_epochs=num_train_epochs,  # Note: sft.py uses 'num_epochs', not 'num_train_epochs'
            learning_rate=learning_rate,
            cutoff_len=512,  # Aligned with sft.py
            
            # Model settings
            train_from_scratch=train_from_scratch,
            freeze_LLM=freeze_LLM,
            group_by_length=False,  # Aligned with sft.py default
            
            # Category
            category=category,
            
            # Index and metadata (aligned with sft.py signature)
            sid_index_path=sid_index_path if sid_index_path else os.path.join(onerec_base, f"{data_name}.index.json"),
            item_meta_path=item_meta_path if item_meta_path else os.path.join(onerec_base, f"{data_name}.item.json"),
            
            # Wandb
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name if wandb_run_name else f"{data_name}_cycle_{cycle}",
            
            # Resume from checkpoint (if applicable)
            resume_from_checkpoint=None,
        )
        
        print("\n" + "=" * 60)
        print(f"✓ Training completed! Checkpoint saved to: {cycle_output_dir}")
        print("=" * 60)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("GPU memory cleared")
        
        return cycle_output_dir
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    fire.Fire(train_simulation)

