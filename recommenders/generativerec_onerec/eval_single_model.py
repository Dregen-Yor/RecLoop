"""
OneRec Single Model Evaluation Script (Single Cycle Evaluation)
Used to evaluate OneRec models trained via sft_simulation.py

Evaluation process:
1. Find test files and info files
2. Call evaluate.py for inference
3. Call calc.py to calculate metrics
"""

import os
import sys
import glob
import fire
import subprocess
from pathlib import Path


def eval_single_model(
    model_path: str,
    data_dir: str,
    data_name: str,
    cycle: int,
    output_dir: str = None,
    gpu_id: str = "0",
    batch_size: int = 8,
    num_beams: int = 50,
    max_new_tokens: int = 256,
    length_penalty: float = 0.0,
    seed: int = 42,
):
    """
    Evaluate retrieval metrics for OneRec model (single cycle evaluation)
    
    Args:
        model_path: Model checkpoint path
        data_dir: Simulation data directory
        data_name: Dataset name, e.g., "Toys_and_Games"
        cycle: Evaluation cycle
        output_dir: Custom output directory (optional, default: script_dir/results/model_name)
        gpu_id: GPU ID to use, default "0"
        batch_size: Batch size, default 8
        num_beams: Beam search count, default 50
        max_new_tokens: Maximum generated tokens, default 256
        length_penalty: Length penalty, default 0.0
        seed: Random seed, default 42
    """
    
    if not model_path or not data_dir or not data_name or cycle is None:
        raise ValueError("Required parameters: model_path, data_dir, data_name, cycle")
    
    print("=" * 80)
    print("OneRec Single-Cycle Evaluation Script")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Dataset: {data_name}")
    print(f"Cycle: cycle-{cycle}")
    print(f"GPU: {gpu_id}")
    print("=" * 80)
    
    return _eval_one_cycle(
        model_path=model_path,
        data_dir=data_dir,
        data_name=data_name,
        cycle=cycle,
        output_dir=output_dir,
        gpu_id=gpu_id,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        seed=seed
    )


def _eval_one_cycle(
    model_path: str,
    data_dir: str,
    data_name: str,
    cycle: int,
    output_dir: str = None,
    gpu_id: str = "0",
    batch_size: int = 8,
    num_beams: int = 50,
    max_new_tokens: int = 256,
    length_penalty: float = 0.0,
    seed: int = 42,
):
    """
    Internal function for evaluating a single cycle
    """
    
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
        print(f"✓ Model path validation passed")
    
    test_dirs = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "test"),
    ]
    
    test_file = None
    for test_dir in test_dirs:
        test_pattern = os.path.join(test_dir, f"{data_name}_cycle_{cycle}_test.csv")
        test_files = glob.glob(test_pattern)
        if test_files:
            test_file = test_files[0]
            break
    
    if not test_file:
        raise FileNotFoundError(
            f"Test file not found: {data_name}_cycle_{cycle}_test.csv\n"
            f"Searched directories:\n" +
            "\n".join([f"  - {d}" for d in test_dirs])
        )
    
    test_file = os.path.abspath(test_file)
    print(f"✓ Found test file: {test_file}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    info_search_paths = [
        os.path.join(script_dir, "data", data_name, "info"),
        os.path.join(project_root, "recommenders", "generativerec_onerec", "data", data_name, "info"),
        os.path.join(data_dir, "..", "data", data_name, "info"),
    ]
    
    info_file = None
    for search_path in info_search_paths:
        if os.path.exists(search_path):
            info_files = glob.glob(os.path.join(search_path, f"{data_name}*.txt"))
            if info_files:
                info_file = info_files[0]
                break
    
    if not info_file:
        raise FileNotFoundError(
            f"Info file not found, searched the following locations:\n" + 
            "\n".join([f"  - {p}" for p in info_search_paths])
        )
    
        print(f"✓ Found info file: {info_file}")
    
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books"
    }
    category = category_dict.get(data_name, data_name)
    
    if output_dir is None:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(script_dir, "results", model_name)
    else:
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    print("\n" + "=" * 80)
    print("Starting inference evaluation...")
    print("=" * 80)
    
    result_json = os.path.abspath(os.path.join(output_dir, f"result_{data_name}_cycle_{cycle}.json"))
    
    evaluate_cmd = [
        sys.executable,
        os.path.join(script_dir, "evaluate.py"),
        "--base_model", model_path,
        "--info_file", info_file,
        "--category", data_name,
        "--test_data_path", test_file,
        "--result_json_data", result_json,
        "--batch_size", str(batch_size),
        "--num_beams", str(num_beams),
        "--max_new_tokens", str(max_new_tokens),
        "--length_penalty", str(length_penalty),
        "--seed", str(seed),
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Command: {' '.join(evaluate_cmd)}")
    print(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
    
    try:
        subprocess.run(
            evaluate_cmd,
            env=env,
            check=True,
            cwd=script_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Inference evaluation failed: {e}")
        raise
    
        print(f"\n✓ Inference completed, results saved to: {result_json}")
    
    print("\n" + "=" * 80)
    print("Calculating evaluation metrics...")
    print("=" * 80)
    
    calc_cmd = [
        sys.executable,
        os.path.join(script_dir, "calc.py"),
        "--path", result_json,
        "--item_path", info_file,
    ]
    
    print(f"Command: {' '.join(calc_cmd)}\n")
    
    try:
        subprocess.run(
            calc_cmd,
            check=True,
            cwd=script_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Metric calculation failed: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("✓ Evaluation completed!")
    print("=" * 80)
    print(f"Result file: {result_json}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return result_json


if __name__ == '__main__':
    fire.Fire(eval_single_model)
