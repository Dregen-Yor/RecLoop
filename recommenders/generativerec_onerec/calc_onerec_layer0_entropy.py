#!/usr/bin/env python3
"""
calc_onerec_layer0_entropy.py - OneRec layer 0 semantic code entropy calculation script
Based on recommend_gen_onerec.py usage and parameters, compute layer 0 code distribution and entropy
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def ensure_directory(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


class Tokenizer:
    """Tokenizer wrapper class (consistent with recommend_gen_onerec.py)"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


def get_history(history_sids: List[str]) -> Dict[str, str]:
    """Get history information (consistent with recommend_gen_onerec.py)"""
    if not history_sids:
        history = ""
    else:
        L = len(history_sids)
        history = ""
        for i in range(L):
            if i == 0:
                history += history_sids[i]
            else:
                history += ", " + history_sids[i]

    return {
        "input": f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
        "output": ""
    }


def generate_prompt(data_point: Dict[str, str]) -> str:
    """Generate prompt (consistent with recommend_gen_onerec.py)"""
    return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


class OneRecLayer0EntropyCalculator:
    """
    Load trained OneRec model, compute layer 0 code original probability distribution and entropy
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.cycle = args.cycle
        self.batch_size = args.batch_size
        self.codebook_size = args.codebook_size

        self.layer0_prob_accum = np.zeros(self.codebook_size, dtype=np.float64)
        self.layer0_prob_user_count: int = 0

        self.layer0_token_ids: List[int] = []
        self.layer0_token_to_idx: Dict[int, int] = {}

        self.model = None
        self.tokenizer = None
        self.tokenizer_obj = None

        self.itemid_to_sid: Dict[str, str] = {}

        self._load_model()
        self._load_sid_mappings()
        self._build_layer0_token_mapping()

    def _load_model(self) -> None:
        """Load OneRec model and tokenizer"""
        print(f"Running: Loading OneRec model from: {self.args.model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        self.tokenizer_obj = Tokenizer(self.tokenizer)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id

        self.model.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            do_sample=False,
            temperature=1.0,
            top_k=None,
            top_p=None,
        )

        print(f"✓ Model loaded on: {self.device}")

    def _load_sid_mappings(self) -> None:
        """Load SID mappings"""
        print(f"Running: Loading SID mappings from: {self.args.index_file}")

        with open(self.args.index_file, 'r') as f:
            item_to_sid_tokens = json.load(f)

        self.itemid_to_sid = {
            item_id: ''.join(tokens)
            for item_id, tokens in item_to_sid_tokens.items()
        }

        print(f"✓ Loaded {len(self.itemid_to_sid)} item-to-SID mappings")

    def _build_layer0_token_mapping(self) -> None:
        """
        Build layer 0 tokens to vocabulary ID mapping
        Layer 0 tokens are <a_0> to <a_{codebook_size-1}>
        """
        print("Running: Building layer 0 token mapping...")

        layer0_tokens = [f"<a_{i}>" for i in range(self.codebook_size)]

        for idx, token in enumerate(layer0_tokens):

            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = encoded[0]
                self.layer0_token_ids.append(token_id)
                self.layer0_token_to_idx[token_id] = idx
            else:

                pass

        print(f"✓ Found {len(self.layer0_token_ids)} layer 0 tokens")
        if len(self.layer0_token_ids) > 0:
            print(f"  Token ID range: {min(self.layer0_token_ids)} to {max(self.layer0_token_ids)}")

    def _prepare_batch_input(
        self,
        user_data_list: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Prepare batch input (consistent with recommend_gen_onerec.py)

        Args:
            user_data_list: list of dictionaries containing user_id and history_item_sid

        Returns:
            (input_ids, attention_mask, prompt_len)
        """
        encodings = []

        for user_data in user_data_list:

            history_sids = user_data.get('history_item_sid', [])
            if not isinstance(history_sids, list):
                history_sids = []

            instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

"""
            tokens = self.tokenizer_obj.encode(instruction, bos=True, eos=False)

            history = get_history(history_sids)
            prompt = generate_prompt(history)
            tokens = tokens + self.tokenizer_obj.encode(prompt, bos=False, eos=False)

            attention_mask = [1] * len(tokens)

            encodings.append({
                "input_ids": tokens,
                "attention_mask": attention_mask,
            })

        max_len = max([len(_["input_ids"]) for _ in encodings])

        padding_input_ids = []
        padding_attention_mask = []

        for enc in encodings:
            L = len(enc["input_ids"])
            padding_input_ids.append(
                [self.tokenizer.pad_token_id] * (max_len - L) + enc["input_ids"]
            )
            padding_attention_mask.append(
                [0] * (max_len - L) + [1] * L
            )

        input_ids = torch.tensor(padding_input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(padding_attention_mask, dtype=torch.long, device=self.device)

        return input_ids, attention_mask, max_len

    def _extract_layer0_probabilities(
        self, logits: torch.Tensor
    ) -> Optional[np.ndarray]:
        """
        Extract layer 0 semantic code probability distribution from logits

        Correct approach:
        1. Get logits from last position (about to generate first token)
        2. Extract logits corresponding to layer 0 tokens
        3. Compute softmax only on these logits for normalized probabilities
        4. Average across batch

        Args:
            logits: model output logits, shape [batch_size, seq_len, vocab_size]

        Returns:
            Average probability distribution across batch, shape [codebook_size]
        """
        if logits is None or logits.numel() == 0:
            return None

        last_logits = logits[:, -1, :]

        if len(self.layer0_token_ids) == 0:
            return None

        layer0_logits = last_logits[:, self.layer0_token_ids]

        layer0_probs = torch.softmax(layer0_logits, dim=-1)

        mean_probs = layer0_probs.mean(dim=0)

        full_probs = np.zeros(self.codebook_size, dtype=np.float64)

        probs_np = mean_probs.detach().cpu().float().numpy()

        for i, token_id in enumerate(self.layer0_token_ids):
            idx = self.layer0_token_to_idx[token_id]
            full_probs[idx] = probs_np[i]

        return full_probs

    def _collect_layer0_probabilities(
        self, probs: Optional[np.ndarray], batch_size: int
    ) -> None:
        """
        Accumulate layer 0 probability distribution

        Args:
            probs: batch average probability distribution
            batch_size: actual number of users in this batch
        """
        if probs is None:
            return

        self.layer0_prob_accum += probs * batch_size
        self.layer0_prob_user_count += batch_size

    def predict_for_batch(
        self,
        user_data_list: List[Dict],
    ) -> Optional[np.ndarray]:
        """
        Batch predict and extract layer 0 probability distribution

        Args:
            user_data_list: list of user data

        Returns:
            Layer 0 probability distribution averaged across batch
        """
        if not user_data_list:
            return None

        input_ids, attention_mask, _ = self._prepare_batch_input(user_data_list)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        return self._extract_layer0_probabilities(outputs.logits)

    def calculate_entropy(self) -> float:
        """Calculate entropy of accumulated probability distribution"""
        if self.layer0_prob_user_count == 0:
            return 0.0
        mean_probs = self.layer0_prob_accum / self.layer0_prob_user_count
        total_prob = float(mean_probs.sum())
        if total_prob <= 0:
            return 0.0
        normalized_probs = mean_probs / total_prob
        entropy = 0.0
        for prob in normalized_probs:
            if prob > 0:
                entropy -= float(prob * np.log2(prob))
        return float(entropy)

    def predict_batch(
        self,
        user_sequences_dict: Dict[int, List[int]],
    ) -> Dict[str, object]:
        """
        Process all users in batch, compute layer 0 probability distribution

        Args:
            user_sequences_dict: {user_id: sequence} dictionary

        Returns:
            Dictionary containing entropy statistics
        """

        inference_data = []
        for user_id, items in user_sequences_dict.items():

            history_item_sids = []
            for item_id in items:
                if str(item_id) in self.itemid_to_sid:
                    history_item_sids.append(self.itemid_to_sid[str(item_id)])

            if history_item_sids:
                inference_data.append({
                    'user_id': user_id,
                    'history_item_sid': history_item_sids
                })

        print(f"\nStarting layer 0 statistics for {len(inference_data)} users...")
        print(f"  Batch size: {self.batch_size}")

        num_batches = (len(inference_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(inference_data))

            batch_data = inference_data[start_idx:end_idx]
            batch_user_ids = [d['user_id'] for d in batch_data]

            try:
                layer0_probs = self.predict_for_batch(batch_data)
                actual_batch_size = len(batch_data)
                self._collect_layer0_probabilities(layer0_probs, actual_batch_size)

                print(
                    f"  Progress: {end_idx}/{len(inference_data)} - "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"(users {batch_user_ids[0]} to {batch_user_ids[-1]})"
                )
            except Exception as exc:
                print(
                    f"  ⚠️ Batch {batch_idx + 1} processing failed "
                    f"(users {batch_user_ids[0]} to {batch_user_ids[-1]}): {exc}"
                )

        entropy = self.calculate_entropy()
        max_entropy = float(np.log2(len(self.layer0_token_ids))) if len(self.layer0_token_ids) > 0 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        layer0_prob_distribution: Dict[str, float] = {}
        if self.layer0_prob_user_count > 0:
            mean_probs = self.layer0_prob_accum / self.layer0_prob_user_count
            layer0_prob_distribution = {
                str(idx): float(prob) for idx, prob in enumerate(mean_probs)
            }

        return {
            "cycle": self.cycle,
            "num_users": len(inference_data),
            "codebook_size": self.codebook_size,
            "actual_layer0_tokens": len(self.layer0_token_ids),
            "batch_size": self.batch_size,
            "layer0_entropy": entropy,
            "layer0_max_possible_entropy": max_entropy,
            "layer0_normalized_entropy": normalized_entropy,
            "layer0_prob_distribution": layer0_prob_distribution,
        }


def load_user_sequences(
    data_path: str, start_user: int = 1, end_user: int = None
) -> Dict[int, List[int]]:
    """Load user sequence data (format consistent with Tiger version)"""
    user_sequences: Dict[int, List[int]] = {}
    with open(data_path, 'r', encoding='utf-8') as f:
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
            except ValueError as exc:
                print(f"⚠️ Line {line_num} parsing failed: {exc}")
                continue
    return user_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="OneRec layer 1 semantic code entropy calculation script")

    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to user sequence data file')
    parser.add_argument('--index_file', type=str, required=True, help='Path to index.json file (item_id to SID mapping)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to entropy statistics output')

    parser.add_argument('--start_user', type=int, default=1, help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=None, help='Ending user ID')

    parser.add_argument('--batch_size', type=int, default=8, help='Processing batch size')
    parser.add_argument('--codebook_size', type=int, default=256, help='RQ-VAE codebook size')

    parser.add_argument('--cycle', type=int, default=1, help='Current cycle number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80)
    print("📊 OneRec Layer 1 Semantic Code Entropy Calculation")
    print("=" * 80)
    print(f"  - Model path: {args.model_path}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Index file: {args.index_file}")
    print(f"  - Output path: {args.output_path}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Current cycle: {args.cycle}")
    print("=" * 80 + "\n")

    print("Running: Loading user data...")
    user_sequences = load_user_sequences(
        args.data_path, args.start_user, args.end_user
    )
    print(f"✓ Loaded {len(user_sequences)} users' data\n")

    if not user_sequences:
        print("❌ Error: No user sequence data found")
        return

    print("Running: Computing OneRec entropy...")
    calculator = OneRecLayer0EntropyCalculator(args)
    print()

    results = calculator.predict_batch(user_sequences)

    ensure_directory(os.path.dirname(args.output_path))
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"\n✓ Entropy statistics saved to: {args.output_path}")

    print("\n" + "=" * 80)
    print("📈 Statistics Results Summary")
    print("=" * 80)
    print(f"  - Processed users: {results['num_users']}")
    print(f"  - Codebook size: {results['codebook_size']}")
    print(f"  - Layer 0 tokens: {results['actual_layer0_tokens']}")
    print(f"  - Layer 0 entropy: {results['layer0_entropy']:.4f}")
    print(f"  - Max possible entropy: {results['layer0_max_possible_entropy']:.4f}")
    print(f"  - Normalized entropy: {results['layer0_normalized_entropy']:.4f}")
    print("=" * 80 + "\n")
    print("✅ Statistics complete")


if __name__ == "__main__":
    main()
