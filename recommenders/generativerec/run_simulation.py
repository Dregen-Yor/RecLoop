# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
run_simulation.py - simulation loop system training script based on run.py
Minimum modification principle: only make necessary changes to support simulation loop
"""

import os
import sys
import traceback
import argparse

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from src.training import train_tiger
from utils import cf_embedding_path, set_seed


def create_config(args):
    """
    Create config using Hydra, and apply runtime overrides
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="main")
    OmegaConf.set_struct(config, False)

    OmegaConf.update(config, "dataset.name", args.data_name, merge=True)
    OmegaConf.update(
        config,
        "dataset.type",
        "steam" if args.dataset_type == "steam" else "Amazon",
        merge=True,
    )
    OmegaConf.update(config, "dataset.content_model", args.content_model, merge=True)
    OmegaConf.update(config, "dataset.text_emb", args.text_emb, merge=True)
    OmegaConf.update(config, "dataset.cf_model", args.cf_model, merge=True)
    OmegaConf.update(
        config, "dataset.max_items_per_seq", args.max_items_per_seq, merge=True
    )
    OmegaConf.update(config, "max_items_per_seq", args.max_items_per_seq, merge=True)
    OmegaConf.update(
        config, "dataset.features_needed", args.features_needed.split(","), merge=True
    )
    OmegaConf.update(config, "dataset.prompt_format", args.prompt_format, merge=True)
    OmegaConf.update(config, "dataset.raw_data_path", args.data_dir, merge=True)
    OmegaConf.update(config, "dataset.processed_data_path", args.data_dir, merge=True)

    OmegaConf.update(
        config, "method.include_user_id", args.include_user_id, merge=True
    )
    OmegaConf.update(
        config,
        "method.flag_use_learnable_text_embed",
        args.flag_use_learnable_text_embed,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.flag_use_output_embedding",
        args.flag_use_output_embedding,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.text_embedding_dim",
        args.text_embedding_dim,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.hidden_sizes",
        [int(x) for x in args.hidden_sizes.split(",")],
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.embed_proj_type",
        args.embed_proj_type,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.embd_proj_in_dropout_rate",
        args.embd_proj_in_dropout_rate,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.embd_proj_dropout_rate",
        args.embd_proj_dropout_rate,
        merge=True,
    )
    OmegaConf.update(
        config,
        "method.embedding_head_dict.use_new_init",
        args.use_new_init,
        merge=True,
    )

    OmegaConf.update(config, "test_method", "tiger", merge=True)
    OmegaConf.update(config, "seed", args.seed, merge=True)
    OmegaConf.update(config, "device_id", args.device_id, merge=True)
    OmegaConf.update(config, "experiment_id", f"cycle_{args.cycle}", merge=True)
    OmegaConf.update(config, "logging.writer", args.log_writer, merge=True)
    OmegaConf.update(config, "logging.mode", args.log_mode, merge=True)
    OmegaConf.update(config, "logging.project", "tiger_simulation", merge=True)

    max_items_per_seq = OmegaConf.select(config, "max_items_per_seq")
    include_user_id = OmegaConf.select(config, "method.include_user_id")
    n_positions = OmegaConf.select(config, "dataset.TIGER.n_positions")
    if max_items_per_seq is not None:
        n_codebook = 4
        required_n_positions = (
            (int(max_items_per_seq) - 1) * n_codebook
            + (1 if include_user_id else 0)
            + 1
        )
        if n_positions is None:
            new_n_positions = required_n_positions
        else:
            new_n_positions = max(int(n_positions), required_n_positions)
        OmegaConf.update(config, "dataset.TIGER.n_positions", new_n_positions, merge=True)

    config["output_path"] = os.path.join(
        args.output_dir,
        f"{args.data_name}_cycle_{args.cycle}"
    )
    os.makedirs(config["output_path"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser(description="TIGER simulation loop training script")

    parser.add_argument('--cycle', type=int, required=True, help='Current cycle number')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--dataset_type', type=str, default='amazon', 
                        choices=['amazon', 'steam'], help='Dataset type')
    parser.add_argument('--content_model', type=str, default='sentence-t5-xxl', 
                        help='Content model name')
    parser.add_argument('--text_emb', type=bool, default=False, help='Whether to use text embedding')
    parser.add_argument('--cf_model', type=str, default='SASRec', help='CF model name')
    parser.add_argument('--max_items_per_seq', type=int, default=20, 
                        help='Maximum number of items per sequence')
    parser.add_argument('--features_needed', type=str, default='title,brand,price', 
                        help='Comma separated list of needed features')
    parser.add_argument('--prompt_format', type=str, default='noprompt', 
                        help='Prompt format')

    parser.add_argument('--include_user_id', action='store_true', default=False,
                        help='Whether to include user ID')
    parser.add_argument('--flag_use_learnable_text_embed', action='store_true', default=True,
                        help='Whether to use learnable text embedding')
    parser.add_argument('--flag_use_output_embedding', action='store_true', default=False,
                        help='Whether to use output embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=4096,
                        help='Text embedding dimension')
    parser.add_argument('--hidden_sizes', type=str, default='4096,2048',
                        help='MLP hidden sizes, comma separated')
    parser.add_argument('--embed_proj_type', type=str, default='mlp',
                        choices=['mlp', 'linear'], help='Embedding projection type')
    parser.add_argument('--embd_proj_in_dropout_rate', type=float, default=0.1,
                        help='Embedding projection input dropout rate')
    parser.add_argument('--embd_proj_dropout_rate', type=float, default=0.1,
                        help='Embedding projection dropout rate')
    parser.add_argument('--use_new_init', action='store_true', default=False,
                        help='Whether to use new initialization')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128,
                        help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=256,
                        help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Logging steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--eval_accumulation_steps', type=int, default=1,
                        help='Evaluation accumulation steps')

    parser.add_argument('--log_writer', type=str, default='wandb',
                        choices=['wandb', 'tensorboard'], help='Log writer type')
    parser.add_argument('--log_mode', type=str, default='disabled',
                        choices=['online', 'offline', 'disabled'], help='Log mode')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 Starting TIGER Training")
    print("=" * 80)
    print(f"  - Dataset: {args.data_name}")
    print(f"  - Cycle: {args.cycle}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")

    print(f"\n[ Step 1 ] Setting random seed...")
    device = (
        torch.device(f"cuda:{args.device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"  ✓ Using device: {device}")
    set_seed(args.seed)
    print(f"  ✓ Random seed: {args.seed}")

    print(f"\n[ Step 2 ] Creating configuration...")
    config = create_config(args)
    is_steam = str(config["dataset"]["type"]).lower() == "steam"
    print(f"  ✓ Output path: {config['output_path']}")
    print(f"  ✓ Dataset type: {'Steam' if is_steam else 'Amazon'}")

    try:
        print(f"\n[ Step 3 ] Checking data files...")
        features_used = "_".join(config["dataset"]["features_needed"])

        data_file = os.path.join(args.data_dir, f"{args.data_name}-{args.cycle}.txt")
        id2meta_filename = (
            f"{args.data_name}_{features_used}_{args.prompt_format}_id2meta.json"
        )
        id2meta_file = os.path.join(args.data_dir, id2meta_filename)
        fallback_id2meta_file = os.path.join(
            args.data_dir, args.data_name, id2meta_filename
        )
        item2attribute_file = os.path.join(
            args.data_dir, f"{args.data_name}_item2attributes.json"
        )

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if not os.path.exists(id2meta_file) and os.path.exists(fallback_id2meta_file):
            id2meta_file = fallback_id2meta_file
        if not os.path.exists(id2meta_file):
            raise FileNotFoundError(f"Metadata file not found: {id2meta_file}")

        print(f"  ✓ Found data files")
        print(f"    - Interaction sequence: {data_file}")
        print(f"    - Metadata: {id2meta_file}")

        print(f"\n[ Step 4 ] Loading data...")
        print(f"    - Data file: {data_file}")
        print(f"    - Reading metadata: {id2meta_file}")
        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )
        print(f"  ✓ Data loading complete")
        print(f"    - User count: {len(user_sequence)}")
        print(f"    - Seen items: {len(id_split['seen'])}")
        print(f"    - Unseen val items: {len(id_split['unseen_val'])}")
        print(f"    - Unseen test items: {len(id_split['unseen_test'])}")

        print(f"\n[ Step 5 ] Loading embeddings...")
        if args.text_emb is True:
            id_filename = f"{args.data_name}_{args.content_model}"
            embedding_save_path = os.path.join(
                os.path.join(args.data_dir, args.data_name), f"{id_filename}_embeddings.pt"
            )
            print(f"    - Embedding save path: {embedding_save_path}")
            item_embedding = process_embeddings(
                config, device, id2meta_file, embedding_save_path
            )
        else:
            id_filename = f"{args.data_name}_{args.cf_model}"
            embedding_save_path = cf_embedding_path(args.cf_model, args.data_name)
            print(f"    - Embedding save path: {embedding_save_path}")
            item_embedding = torch.load(embedding_save_path, weights_only=False).to(device)

        print(f"  ✓ Embedding loading complete")
        print(f"    - Embedding shape: {item_embedding.shape}")

        print(f"\n[ Step 6 ] Checking RQ-VAE...")
        id_save_location = os.path.join(
            args.data_dir, f"{id_filename}_{args.seed}.pkl"
        )
        if not os.path.exists(id_save_location):
            print(f"    RQ-VAE file not found: {id_save_location}")
            train_sid(
                config, device, item_embedding, id_split, id_save_location
            )
        else:
            print(f"  ✓ Found RQ-VAE: {id_save_location}")

        print(f"  ✓ RQ-VAE ready: {id_save_location}")

        print(f"\n[ Step 7 ] Training TIGER model...")
        print(f"    - Output path: {config['output_path']}")
        train_tiger(
            config,
            train_config={**config["dataset"], **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]}},
            method_config={**config["method"], **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]}},
            id_split=id_split,
            user_sequence=user_sequence,
            item_embedding=item_embedding,
            id_save_location=id_save_location,
            device=device,
        )
        print(f"  ✓ TIGER model training complete")

        print("\n" + "=" * 80)
        print("🎉 Training complete! Simulation loop finished")
        print("=" * 80 + "\n")

    except BaseException as e:
        print("\n" + "=" * 80)
        print(f"❌ Error: Training failed")
        print("=" * 80)
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        # Flush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
