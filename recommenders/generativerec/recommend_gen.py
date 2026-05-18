"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

recommend_gen.py - TIGER recommendation generation script
Adapted for liger (generativerec) version, supports simulation loop
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import T5Config

from src.tiger import TIGER
from src.load_data import expand_id, get_unique_semantic_ids_by_extra_position


def ensure_directory(path: Optional[str]) -> None:
    """Create directory if it does not exist"""
    if path:
        os.makedirs(path, exist_ok=True)


class TIGERRecommendationGenerator:
    """
    Load trained TIGER model and generate top-K recommendations
    Decode semantic codes back to item IDs using beam search
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.recommendation_history_file = getattr(
            args, "recommendation_history_file", "./recommendation_history.txt"
        )
        self.exclude_recommended = getattr(args, "exclude_recommended", True)
        self.cycle = args.cycle
        self.beam_size = args.beam_size
        self.max_items_per_seq = args.max_items_per_seq
        self.codebook_size = args.codebook_size

        self.item_to_semantic_id: Dict[int, Tuple[int, ...]] = {}
        self.semantic_id_to_item: Dict[Tuple[int, ...], int] = {}
        self.n_semantic_codebook: int = 3
        self.n_codebook: int = 4
        self.max_last_semantic_ids: int = 0
        self.vocab_size: int = 0
        self.max_item_id: int = 0

        self.model: Optional[TIGER] = None

        self._load_semantic_ids()
        self._load_model()

    def _load_semantic_ids(self) -> None:
        """Load RQ-VAE semantic ID codebook"""
        if not os.path.exists(self.args.semantic_id_path):
            raise FileNotFoundError(
                f"Semantic ID file not found: {self.args.semantic_id_path}"
            )

        print(f"Running: Loading semantic IDs from: {self.args.semantic_id_path}")
        semantic_ids = pickle.load(open(self.args.semantic_id_path, "rb"))

        self.item_to_semantic_id, self.max_last_semantic_ids = (
            get_unique_semantic_ids_by_extra_position(semantic_ids, self.codebook_size)
        )

        for item_id, semantic_id in self.item_to_semantic_id.items():
            expanded = tuple(expand_id(semantic_id, self.codebook_size))
            self.semantic_id_to_item[expanded] = item_id

        self.max_item_id = max(self.item_to_semantic_id.keys())
        last_codebook_size = max(self.max_last_semantic_ids, self.codebook_size)

        if self.args.include_user_id:
            self.vocab_size = (
                2000
                + self.codebook_size * self.n_semantic_codebook
                + last_codebook_size
                + 2
            )
        else:
            self.vocab_size = (
                self.codebook_size * self.n_semantic_codebook
                + last_codebook_size
                + 2
            )

        print(
            f"✓ Loaded semantic IDs for {len(self.item_to_semantic_id)} items"
        )
        print(f"  - Max item ID: {self.max_item_id}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Codebooks: {self.n_codebook}")

    def _load_model(self) -> None:
        """Load TIGER model and checkpoint"""
        model_path = os.path.join(self.args.model_path, "results", "ckpt_best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.args.model_path, "ckpt_best.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Running: Loading TIGER model from: {model_path}")

        t5_config = T5Config(
            num_layers=self.args.num_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            d_model=self.args.d_model,
            d_ff=self.args.d_ff,
            num_heads=self.args.num_heads,
            d_kv=self.args.d_kv,
            dropout_rate=self.args.dropout_rate,
            vocab_size=self.vocab_size,
            pad_token_id=0,
            eos_token_id=int(self.vocab_size - 1),
            decoder_start_token_id=0,
            feed_forward_proj=self.args.feed_forward_proj,
            n_positions=self.args.n_positions,
            layer_norm_epsilon=1e-8,
            initializer_factor=self.args.initializer_factor,
        )

        self.model = TIGER(
            config=t5_config,
            n_semantic_codebook=self.n_semantic_codebook,
            max_items_per_seq=self.max_items_per_seq,
            flag_use_learnable_text_embed=self.args.flag_use_learnable_text_embed,
            flag_use_output_embedding=self.args.flag_use_output_embedding,
            embedding_head_dict=self.args.embedding_head_dict,
        )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded on: {self.device}")

    def _prepare_input(
        self, input_sequence: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert user history sequence to model input

        Args:
            input_sequence: user interaction item ID list

        Returns:
            (input_ids, attention_mask)
        """

        if len(input_sequence) > self.max_items_per_seq:
            input_sequence = input_sequence[-self.max_items_per_seq:]

        input_sids = []
        for item_id in input_sequence:
            if item_id in self.item_to_semantic_id:
                semantic_id = self.item_to_semantic_id[item_id]
                expanded = expand_id(semantic_id, self.codebook_size)
                input_sids.extend(expanded)
            else:
                input_sids.extend([0] * self.n_codebook)

        max_length = self.max_items_per_seq * self.n_codebook

        # Padding
        if len(input_sids) < max_length:
            input_sids = input_sids + [0] * (max_length - len(input_sids))
        else:
            input_sids = input_sids[:max_length]

        input_ids = torch.tensor([input_sids], dtype=torch.long, device=self.device)
        attention_mask = (input_ids != 0).long()

        return input_ids, attention_mask

    def _decode_output(
        self, sequences: torch.Tensor, scores: torch.Tensor, debug: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Decode model generated semantic codes back to item IDs

        Args:
            sequences: generated sequences [beam_size, seq_len]
            scores: beam scores [beam_size]
            debug: whether to print debug info

        Returns:
            [(item_id, probability), ...], sorted descending by probability
        """
        if sequences.numel() == 0:
            if debug:
                print("  [] No generation")
            return []

        if debug:
            print(f"  [] Generated {sequences.size(0)} beams, length={sequences.size(1)}")

        sequences = sequences[:, 1:]

        if sequences.size(1) < self.n_codebook:
            pad_tokens = torch.zeros(
                (sequences.size(0), self.n_codebook - sequences.size(1)),
                device=sequences.device,
                dtype=sequences.dtype,
            )
            sequences = torch.cat([sequences, pad_tokens], dim=1)

        sequences = sequences[:, :self.n_codebook]

        probs = torch.softmax(scores.to(sequences.device), dim=0)

        decoded: Dict[int, float] = {}
        valid_count = 0
        invalid_count = 0
        duplicate_count = 0

        for idx, (codes, prob) in enumerate(zip(sequences.tolist(), probs.tolist())):

            if all(token == 0 for token in codes):
                if debug and idx < 3:
                    print(f"  [] Beam {idx}: padding, skipped")
                invalid_count += 1
                continue

            semantic_id = tuple(codes)
            item_id = self.semantic_id_to_item.get(semantic_id)

            if item_id is None or item_id <= 0:
                if debug and idx < 3:
                    print(f"  [] Beam {idx}: semantic codes {codes} no matching item ID")
                invalid_count += 1
                continue

            if item_id in decoded:
                duplicate_count += 1
                if debug and idx < 3:
                    print(f"  [] Beam {idx}: item {item_id} duplicate (prob={prob:.4f})")
            else:
                valid_count += 1
                if debug and idx < 3:
                    print(f"  [] Beam {idx}: item {item_id} (prob={prob:.4f})")

            decoded[item_id] = max(decoded.get(item_id, 0.0), float(prob))

        if debug:
            print(f"  [] Decoding results: {valid_count} valid, {duplicate_count} duplicates, {invalid_count} invalid")

        return sorted(decoded.items(), key=lambda x: x[1], reverse=True)

    def predict_for_user(
        self,
        input_sequence: List[int],
        k: int = 10,
        exclude_seen: bool = True,
        exclude_recommended_items: Optional[List[int]] = None,
        debug: bool = False,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Generate top-K recommendations for a single user

        Args:
            input_sequence: user history interaction sequence
            k: number of recommendations to generate
            exclude_seen: whether to exclude already interacted items
            exclude_recommended_items: list of previously recommended items to exclude
            debug: whether to print debug info

        Returns:
            (recommendations, probability_map)
        """
        if debug:
            print(f"  [] Input sequence length: {len(input_sequence)}")
            print(f"  [] Recommendation count: {k}, exclude seen: {exclude_seen}")
            if exclude_recommended_items:
                print(f"  [] Items to exclude: {len(exclude_recommended_items)}")

        input_ids, attention_mask = self._prepare_input(input_sequence)

        if debug:
            print(f"  [] Generating with: beam size={self.beam_size}, max new tokens={self.n_codebook}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.beam_size,
                max_new_tokens=self.n_codebook,
                num_return_sequences=self.beam_size,
                pad_token_id=0,
                eos_token_id=int(self.vocab_size - 1),
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences
        beam_scores = outputs.sequences_scores

        decoded_items = self._decode_output(sequences, beam_scores, debug=debug)

        if debug:
            print(f"  [] Decoded {len(decoded_items)} items")

        exclusion: set = set()
        if exclude_seen:
            exclusion.update(input_sequence)
        if exclude_recommended_items:
            exclusion.update(exclude_recommended_items)

        if debug:
            print(f"  [] Total items to exclude: {len(exclusion)}")

        recommendations: List[int] = []
        probability_map: Dict[int, float] = {}
        filtered_count = 0

        for item_id, prob in decoded_items:
            if item_id in exclusion:
                filtered_count += 1
                if debug and filtered_count <= 3:
                    print(f"  [] Item {item_id} excluded, skipped")
                continue
            if item_id not in recommendations:
                recommendations.append(item_id)
                if debug and len(recommendations) <= 3:
                    print(f"  [] Recommended item {item_id} (prob={prob:.4f})")
            probability_map[item_id] = prob
            if len(recommendations) >= k:
                break

        if debug:
            print(f"  [] Final recommendations: {len(recommendations)} items, {filtered_count} filtered out")

        return recommendations, probability_map

    def _load_recommendation_history(self) -> Dict[int, List[int]]:
        """Load recommendation history file"""
        history: Dict[int, List[int]] = {}
        if not self.recommendation_history_file or not os.path.exists(
            self.recommendation_history_file
        ):
            return history

        try:
            with open(self.recommendation_history_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        user_id = int(parts[0])
                    except ValueError:
                        continue
                    items: List[int] = []
                    for item in parts[1:]:
                        try:
                            items.append(int(item))
                        except ValueError:
                            continue
                    if items:
                        history[user_id] = items
            if history:
                print(f"✓ Loaded recommendation history for {len(history)} users")
            return history
        except Exception as exc:
            print(f"⚠️ Failed to load recommendation history: {exc}")
            return {}

    def _save_recommendation_history(self, history: Dict[int, List[int]]) -> None:
        """Save recommendation history to file"""
        if not self.recommendation_history_file:
            return

        directory = os.path.dirname(self.recommendation_history_file)
        ensure_directory(directory)

        try:
            with open(self.recommendation_history_file, "w", encoding="utf-8") as f:
                for user_id in sorted(history.keys()):
                    item_ids = history[user_id]
                    if item_ids:
                        line = f"{user_id} " + " ".join(map(str, item_ids)) + "\n"
                        f.write(line)
            print(
                f"✓ Saved recommendation history for {len(history)} users to: {self.recommendation_history_file}"
            )
        except Exception as exc:
            print(f"⚠️ Failed to save recommendation history: {exc}")

    def predict_batch(
        self,
        user_sequences_dict: Dict[int, List[int]],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        """
        Batch generate recommendations for multiple users

        Args:
            user_sequences_dict: {user_id: [item_ids]}
            k: number of recommendations to generate
            exclude_seen: whether to exclude already interacted items

        Returns:
            {user_id: [recommended_item_ids]}
        """
        recommendation_history = self._load_recommendation_history()

        recommendations: Dict[int, List[int]] = {}
        user_ids = sorted(user_sequences_dict.keys())

        print(f"\nStarting Top-{k} recommendation generation for {len(user_ids)} users...")
        for idx, user_id in enumerate(user_ids, 1):
            sequence = user_sequences_dict[user_id]
            try:
                exclude_items = (
                    recommendation_history.get(user_id, [])
                    if self.exclude_recommended
                    else None
                )
                prior_recommended = list(exclude_items) if exclude_items else []

                debug_mode = (idx <= 5)
                if debug_mode:
                    print(f"\n[ User {user_id} ] Starting recommendation generation...")
                    print(f"[ User {user_id} ] History length: {len(sequence)}")

                rec_items, probability_map = self.predict_for_user(
                    sequence,
                    k=k,
                    exclude_seen=exclude_seen,
                    exclude_recommended_items=exclude_items,
                    debug=debug_mode,
                )
                recommendations[user_id] = rec_items

                if self.exclude_recommended:
                    updated = recommendation_history.get(user_id, [])
                    updated.extend(rec_items)
                    recommendation_history[user_id] = updated

                if idx % 10 == 0 or idx == len(user_ids):
                    print(
                        f"  Progress: {idx}/{len(user_ids)} - User {user_id}: {len(rec_items)} recommendations"
                    )
                    if self.exclude_recommended and prior_recommended:
                        print(f"    Excluded {len(set(prior_recommended))} previously recommended items")

            except Exception as exc:
                print(f"  ⚠️ User {user_id} recommendation failed: {exc}")
                recommendations[user_id] = []

        if self.exclude_recommended:
            self._save_recommendation_history(recommendation_history)

        return recommendations

    def save_recommendations(
        self, recommendations: Dict[int, List[int]], output_path: str
    ) -> None:
        """Save recommendation results and create backup with cycle"""
        ensure_directory(os.path.dirname(output_path))

        with open(output_path, "w", encoding="utf-8") as f:
            for user_id in sorted(recommendations.keys()):
                rec_items = recommendations[user_id]
                if rec_items:
                    line = f"{user_id} " + " ".join(map(str, rec_items)) + "\n"
                    f.write(line)

        print(f"\n✓ Recommendation results saved to: {output_path}")

        backup_path = self._add_cycle_to_filename(output_path)
        with open(backup_path, "w", encoding="utf-8") as f:
            for user_id in sorted(recommendations.keys()):
                rec_items = recommendations[user_id]
                if rec_items:
                    line = f"{user_id} " + " ".join(map(str, rec_items)) + "\n"
                    f.write(line)

        print(f"✓ Recommendation results saved to: {backup_path}")

    def _add_cycle_to_filename(self, file_path: str) -> str:
        """Add current cycle number to filename"""
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{name}_{self.cycle}{ext}"
        return os.path.join(dir_name, new_filename)


def load_user_sequences(
    data_path: str, start_user: int = 1, end_user: int = None
) -> Dict[int, List[int]]:
    """
    Load user sequences from data file

    Args:
        data_path: path to data file (txt format)
        start_user: starting user ID
        end_user: ending user ID (None means all users)

    Returns:
        {user_id: [item_ids]}
    """
    user_sequences: Dict[int, List[int]] = {}

    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                items = list(map(int, line.strip().split()))
                if len(items) < 2:
                    continue

                user_id = items[0]
                sequence = items[1:]

                if end_user is not None:
                    if user_id < start_user or user_id > end_user:
                        continue

                user_sequences[user_id] = sequence

            except ValueError as e:
                print(f"⚠️ Line {line_num} parsing failed: {e}")
                continue

    return user_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="TIGER recommendation generation script")

    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to trained model directory'
    )
    parser.add_argument(
        '--data_path', type=str, required=True, help='Path to user sequence data file'
    )
    parser.add_argument(
        '--semantic_id_path', type=str, required=True, help='Path to RQ-VAE semantic ID file'
    )
    parser.add_argument(
        '--output_path', type=str, required=True, help='Path to save recommendation results'
    )

    parser.add_argument('--start_user', type=int, default=1, help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=None, help='Ending user ID')

    parser.add_argument('--k', type=int, default=5, help='Number of recommendations to generate')
    parser.add_argument(
        '--exclude_seen', action='store_true', default=True, help='Exclude already interacted items'
    )
    parser.add_argument(
        '--exclude_recommended',
        action='store_true',
        default=False,
        help='Exclude previously recommended items',
    )
    parser.add_argument(
        '--recommendation_history_file',
        type=str,
        default='./recommendation_history.txt',
        help='Path to recommendation history file',
    )

    parser.add_argument('--max_items_per_seq', type=int, default=50, help='Maximum items per sequence')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam search beam size')
    parser.add_argument('--codebook_size', type=int, default=256, help='RQ-VAE codebook size')
    parser.add_argument('--include_user_id', action='store_true', default=False, help='Include user ID')

    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1024, help='FF dimension')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Key/Value dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument(
        '--feed_forward_proj', type=str, default='relu', help='Feed forward projection'
    )
    parser.add_argument('--n_positions', type=int, default=258, help='N positions')
    parser.add_argument('--initializer_factor', type=float, default=1.0, help='Initializer factor')

    parser.add_argument(
        '--flag_use_learnable_text_embed',
        action='store_true',
        default=False,
        help='Use learnable text embedding',
    )
    parser.add_argument(
        '--flag_use_output_embedding',
        action='store_true',
        default=False,
        help='Use output embedding',
    )
    parser.add_argument('--text_embedding_dim', type=int, default=4096, help='Text embedding dimension')
    parser.add_argument(
        '--hidden_sizes',
        type=str,
        default='4096,2048',
        help='MLP hidden sizes, comma separated',
    )
    parser.add_argument(
        '--embed_proj_type',
        type=str,
        default='mlp',
        choices=['mlp', 'linear'],
        help='Embedding projection type',
    )
    parser.add_argument(
        '--embd_proj_in_dropout_rate', type=float, default=0.1, help='Embedding projection input dropout'
    )
    parser.add_argument(
        '--embd_proj_dropout_rate', type=float, default=0.1, help='Embedding projection dropout'
    )
    parser.add_argument(
        '--use_new_init', action='store_true', default=False, help='Use new initialization'
    )

    parser.add_argument('--cycle', type=int, default=1, help='Current cycle number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    args.embedding_head_dict = {
        "text_embedding_dim": args.text_embedding_dim,
        "hidden_sizes": [int(x) for x in args.hidden_sizes.split(",")],
        "embed_proj_type": args.embed_proj_type,
        "embd_proj_in_dropout_rate": args.embd_proj_in_dropout_rate,
        "embd_proj_dropout_rate": args.embd_proj_dropout_rate,
        "use_new_init": args.use_new_init,
    }

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80)
    print("🚀 TIGER Recommendation Generation")
    print("=" * 80)
    print(f"  - Model path: {args.model_path}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Semantic ID path: {args.semantic_id_path}")
    print(f"  - Output path: {args.output_path}")
    print(f"  - Recommendation count: {args.k}")
    print(f"  - Current cycle: {args.cycle}")
    if args.exclude_recommended:
        print(f"  - Recommendation history file: {args.recommendation_history_file}")
        print("  - Will exclude previously recommended items")
    print("=" * 80 + "\n")

    print("Running: Loading user data...")
    user_sequences = load_user_sequences(
        args.data_path, args.start_user, args.end_user
    )
    print(f"✓ Loaded {len(user_sequences)} users' data\n")

    if not user_sequences:
        print("❌ Error: No user sequence data found")
        return

    print("Running: TIGER recommendation generation...")
    recommender = TIGERRecommendationGenerator(args)
    print()

    recommendations = recommender.predict_batch(
        user_sequences, k=args.k, exclude_seen=args.exclude_seen
    )

    recommender.save_recommendations(recommendations, args.output_path)

    print("\n" + "=" * 80)
    print("🎉 Recommendation generation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
