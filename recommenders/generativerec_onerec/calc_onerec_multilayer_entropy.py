#!/usr/bin/env python3
"""
calc_onerec_multilayer_entropy.py - OneRec multi-layer semantic code entropy calculation script
Following TIGER multi-layer calculation logic, compute joint probability distribution and entropy for Layer0, Layer1, Layer2
Uses Top-p truncation to optimize path count
"""

import argparse
import json
import os
from functools import partial
import time
from typing import Dict, List, Optional, Tuple


print = partial(print, flush=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def ensure_directory(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def top_p_filter(probs: np.ndarray, top_p: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-p (nucleus) filtering, keep tokens with cumulative probability up to top_p
    
    Args:
        probs: probability distribution, shape [num_tokens]
        top_p: cumulative probability threshold (0-1)
        
    Returns:
        (filtered_indices, filtered_probs) - filtered indices and normalized probabilities
    """

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    

    cumsum = np.cumsum(sorted_probs)
    

    cutoff_idx = np.searchsorted(cumsum, top_p) + 1
    cutoff_idx = min(cutoff_idx, len(sorted_probs))
    

    kept_indices = sorted_indices[:cutoff_idx]
    kept_probs = sorted_probs[:cutoff_idx]
    

    kept_probs = kept_probs / kept_probs.sum()
    
    return kept_indices, kept_probs


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


class OneRecMultilayerEntropyCalculator:
    """
    Calculate joint probability distribution and entropy of OneRec multi-layer semantic codes
    Uses Top-p truncation to optimize path count
    
    OneRec uses 3 layers of semantic codes: <a_X>, <b_X>, <c_X>
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.cycle = args.cycle
        self.batch_size = args.batch_size
        self.codebook_size = args.codebook_size
        self.top_p = args.top_p
        self.n_layers = 3


        self.layer_prefixes = ['a', 'b', 'c']
        

        self.layer_token_ids: List[List[int]] = [[] for _ in range(self.n_layers)]
        self.layer_token_to_idx: List[Dict[int, int]] = [{} for _ in range(self.n_layers)]
        
        self.model = None
        self.tokenizer = None
        self.tokenizer_obj = None
        

        self.itemid_to_sid: Dict[str, str] = {}

        self._load_model()
        self._load_sid_mappings()
        self._build_all_layer_token_mappings()

    def _load_model(self) -> None:
        """Load OneRec model and tokenizer"""
        print(f"Loading OneRec model: {self.args.model_path}")
        
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
        
        print(f"✓ Model loaded on {self.device}")

    def _load_sid_mappings(self) -> None:
        """Load SID mappings"""
        print(f"Loading SID mapping: {self.args.index_file}")
        
        with open(self.args.index_file, 'r') as f:
            item_to_sid_tokens = json.load(f)
        
        self.itemid_to_sid = {
            item_id: ''.join(tokens)
            for item_id, tokens in item_to_sid_tokens.items()
        }
        
        print(f"✓ Loaded {len(self.itemid_to_sid)} item-to-SID mappings")

    def _build_all_layer_token_mappings(self) -> None:
        """
        Build token mappings for all layers
        layer0: <a_0> ~ <a_255>
        layer1: <b_0> ~ <b_255>
        layer2: <c_0> ~ <c_255>
        """
        print("Building multi-layer token mappings...")
        
        for layer_idx, prefix in enumerate(self.layer_prefixes):
            tokens = [f"<{prefix}_{i}>" for i in range(self.codebook_size)]
            
            for idx, token in enumerate(tokens):
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if len(encoded) == 1:
                    token_id = encoded[0]
                    self.layer_token_ids[layer_idx].append(token_id)
                    self.layer_token_to_idx[layer_idx][token_id] = idx
            
            print(f"  Layer {layer_idx} ({prefix}): found {len(self.layer_token_ids[layer_idx])} tokens")
        
        print("✓ Multi-layer token mappings built successfully")

    def _prepare_batch_input(
        self, 
        user_data_list: List[Dict]
    ) -> Tuple[List[List[int]], int]:
        """
        Prepare batch inputs (consistent with recommend_gen_onerec.py)
        
        Returns:
            (encodings_list, max_len) - list of encodings and max length
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
            
            encodings.append(tokens)
        
        max_len = max(len(enc) for enc in encodings)
        
        return encodings, max_len

    def _get_layer_probs(
        self,
        base_input_ids: List[List[int]],
        decoder_prefix_tokens: List[int],
        layer: int,
    ) -> np.ndarray:
        """
        Get conditional probability distribution for specified layer
        
        For Causal LM, we need to append prefix tokens to the end of input sequence,
        then get logits at the last position
        
        Args:
            base_input_ids: base input token IDs (each user's prompt)
            decoder_prefix_tokens: decoder prefix tokens (tokens generated by previous layers)
            layer: current layer (0, 1, 2)
            
        Returns:
            conditional probability distribution for this layer's codebook, shape [codebook_size]
        """
        batch_size = len(base_input_ids)
        

        extended_seqs = []
        for base_seq in base_input_ids:
            extended_seqs.append(base_seq + decoder_prefix_tokens)
        
        # Padding (left padding)
        max_len = max(len(seq) for seq in extended_seqs)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for seq in extended_seqs:
            L = len(seq)
            padded_input_ids.append(
                [self.tokenizer.pad_token_id] * (max_len - L) + seq
            )
            padded_attention_mask.append(
                [0] * (max_len - L) + [1] * L
            )
        
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        

        last_logits = outputs.logits[:, -1, :]
        

        layer_token_ids = self.layer_token_ids[layer]
        
        if len(layer_token_ids) == 0:
            return np.zeros(self.codebook_size, dtype=np.float64)
        

        layer_logits = last_logits[:, layer_token_ids]
        

        layer_probs = torch.softmax(layer_logits, dim=-1)
        

        mean_probs = layer_probs.mean(dim=0)
        

        full_probs = np.zeros(self.codebook_size, dtype=np.float64)
        probs_np = mean_probs.detach().cpu().float().numpy()
        
        for i, token_id in enumerate(layer_token_ids):
            idx = self.layer_token_to_idx[layer][token_id]
            full_probs[idx] = probs_np[i]
        
        return full_probs

    def calculate_multilayer_entropy(
        self, user_sequences_dict: Dict[int, List[int]]
    ) -> Dict[str, object]:
        """
        Calculate multi-layer joint probability distribution and entropy
        
        Uses Top-p truncation to optimize path count
        """

        self._start_time = time.time()
        

        inference_data = []
        for user_id, items in sorted(user_sequences_dict.items()):
            history_item_sids = []
            for item_id in items:
                if str(item_id) in self.itemid_to_sid:
                    history_item_sids.append(self.itemid_to_sid[str(item_id)])
            
            if history_item_sids:
                inference_data.append({
                    'user_id': user_id,
                    'history_item_sid': history_item_sids
                })
        
        print(f"\nStarting multi-layer entropy calculation (Top-p = {self.top_p})...")
        print(f"  Users: {len(inference_data)}")
        print(f"  Batch size: {self.batch_size}")
        

        all_base_inputs: List[List[List[int]]] = []  # List of batches
        
        for batch_idx in range(0, len(inference_data), self.batch_size):
            batch_data = inference_data[batch_idx:batch_idx + self.batch_size]
            encodings, _ = self._prepare_batch_input(batch_data)
            all_base_inputs.append(encodings)
        
        # === Layer 0 ===
        print("\n" + "=" * 60)
        print("📊 Processing Layer 0...")
        print("=" * 60)
        
        layer0_prob_accum = np.zeros(self.codebook_size, dtype=np.float64)
        total_samples = 0
        

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] Start Layer 0: {mem_allocated:.2f} GB")
        
        for batch_idx, batch_inputs in enumerate(all_base_inputs):
            probs = self._get_layer_probs(batch_inputs, [], layer=0)
            layer0_prob_accum += probs * len(batch_inputs)
            total_samples += len(batch_inputs)
            
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                if batch_idx % 10 == 0:
                    print(f"  [GPU memory] Batch {batch_idx}/{len(all_base_inputs)}: {mem_allocated:.2f} GB, reserved {mem_reserved:.2f} GB")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] Layer 0 end: {mem_allocated:.2f} GB")
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] After cache clean: {mem_allocated:.2f} GB")
        
        layer0_probs = layer0_prob_accum / total_samples
        

        layer0_kept_indices, layer0_kept_probs = top_p_filter(layer0_probs, self.top_p)
        print(f"✓ Layer 0: kept {len(layer0_kept_indices)} / {self.codebook_size} tokens (top {self.top_p*100:.0f}%)")
        

        print("  [DEBUG] Layer 0 kept tokens:")
        print(f"  layer_token_ids[0]: {len(self.layer_token_ids[0])}")
        for _ci, (_c0_idx, _c0_prob) in enumerate(zip(layer0_kept_indices, layer0_kept_probs)):
            _c0_idx = int(_c0_idx)
            if _c0_idx < len(self.layer_token_ids[0]):
                _tok_id = self.layer_token_ids[0][_c0_idx]
                _tok_str = self.tokenizer.convert_ids_to_tokens([_tok_id])
                print(f"    [{_ci}] codebook_idx={_c0_idx}, token_id={_tok_id}, token_str={_tok_str}, prob={_c0_prob:.6f}")
            else:
                print(f"    [{_ci}] codebook_idx={_c0_idx} out of range! layer_token_ids[0] = {len(self.layer_token_ids[0])}, token_id=None, prob={_c0_prob:.6f}")
        

        layer0_entropy = -np.sum(layer0_kept_probs * np.log2(layer0_kept_probs + 1e-16))
        
        # === Layer 1 ===
        print("\n" + "=" * 60)
        print("📊 Processing Layer 1 (with Top-p)...")
        print("=" * 60)
        

        all_joint_c0_c1: List[Tuple[int, int, float]] = []  # [(c0, c1, joint_prob), ...]
        
        layer1_total_calls = len(layer0_kept_indices)
        print(f"  [Layer 1] GPU calls: {layer1_total_calls} (one per C0 token)")
        
        for idx, (c0_idx, c0_prob) in enumerate(zip(layer0_kept_indices, layer0_kept_probs)):


            c0_idx = int(c0_idx)
            c0_token_id = self.layer_token_ids[0][c0_idx] if c0_idx < len(self.layer_token_ids[0]) else None
            

            print(f"  [DEBUG Layer1] idx={idx}, c0_idx={c0_idx}, c0_token_id={c0_token_id}, valid={c0_token_id is not None}")
            
            if c0_token_id is None:
                continue
            
            if idx % 10 == 0:
                print(f"  Processing Layer0 Code {c0_idx} ({idx + 1}/{len(layer0_kept_indices)})...")
            

            layer1_cond_accum = np.zeros(self.codebook_size, dtype=np.float64)
            total_samples = 0
            
            for batch_inputs in all_base_inputs:
                probs = self._get_layer_probs(
                    batch_inputs, [c0_token_id], layer=1
                )
                layer1_cond_accum += probs * len(batch_inputs)
                total_samples += len(batch_inputs)
            
            layer1_cond_probs = layer1_cond_accum / total_samples
            

            for c1_idx in range(self.codebook_size):
                if layer1_cond_probs[c1_idx] <= 0:
                    continue
                c1_cond_prob = layer1_cond_probs[c1_idx]
                joint_prob = float(c0_prob * c1_cond_prob)
                if joint_prob > 1e-10:
                    all_joint_c0_c1.append((c0_idx, c1_idx, joint_prob))
        
        print(f"  Found {len(all_joint_c0_c1)} (C0, C1) pairs")
        

        all_joint_c0_c1.sort(key=lambda x: x[2], reverse=True)
        joint_probs_array = np.array([x[2] for x in all_joint_c0_c1])
        cumsum = np.cumsum(joint_probs_array)
        total_prob = cumsum[-1] if len(cumsum) > 0 else 0
        
        if total_prob > 0:
            cumsum_normalized = cumsum / total_prob
        else:
            cumsum_normalized = cumsum
        
        cutoff_idx = np.searchsorted(cumsum_normalized, self.top_p) + 1
        cutoff_idx = min(cutoff_idx, len(all_joint_c0_c1))
        
        kept_c0_c1 = all_joint_c0_c1[:cutoff_idx]
        kept_total = sum(x[2] for x in kept_c0_c1)
        

        joint_c1_c2: Dict[int, Dict[int, float]] = {}
        layer1_marginal = np.zeros(self.codebook_size, dtype=np.float64)
        
        for c0_idx, c1_idx, joint_prob in kept_c0_c1:
            normalized_prob = joint_prob / kept_total
            if c0_idx not in joint_c1_c2:
                joint_c1_c2[c0_idx] = {}
            joint_c1_c2[c0_idx][c1_idx] = normalized_prob
            layer1_marginal[c1_idx] += normalized_prob
        

        valid_probs = layer1_marginal[layer1_marginal > 0]
        layer1_entropy = -np.sum(valid_probs * np.log2(valid_probs))
        
        total_paths_c1_c2 = len(kept_c0_c1)
        print(f"✓ Layer 1: top-{self.top_p*100:.0f}% kept {total_paths_c1_c2} paths")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] Layer 1 end: {mem_allocated:.2f} GB")
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] After cache clean: {mem_allocated:.2f} GB")
        
        # === Layer 2 ===
        print("\n" + "=" * 60)
        print("📊 Processing Layer 2 (with Top-p)...")
        print("=" * 60)
        

        all_joint_c0_c1_c2: List[Tuple[int, int, int, float]] = []
        

        total_paths = sum(len(c1_dict) for c1_dict in joint_c1_c2.values())
        print(f"  [Layer 2] GPU calls: {total_paths} (one per (C0, C1) pair)")
        print(f"  [Comparison] Layer 1 GPU calls: {len(layer0_kept_indices)} | Layer 2: {total_paths} | ratio: {total_paths/len(layer0_kept_indices):.1f}x")
        
        path_count = 0
        for c0_idx, c1_dict in joint_c1_c2.items():
            for c1_idx, joint_c0_c1_prob in c1_dict.items():
                path_count += 1

                if path_count % 5 == 0 or path_count == total_paths:
                    elapsed = time.time() - self._start_time if hasattr(self, '_start_time') else 0
                    eta = (elapsed / path_count) * (total_paths - path_count) if path_count > 0 and elapsed > 0 else 0
                    print(f"  Processing: {path_count}/{total_paths} ({path_count/total_paths*100:.1f}%) ETA: {eta:.1f}s")
                

                c0_token_id = self.layer_token_ids[0][c0_idx] if c0_idx < len(self.layer_token_ids[0]) else None
                c1_token_id = self.layer_token_ids[1][c1_idx] if c1_idx < len(self.layer_token_ids[1]) else None
                
                if c0_token_id is None or c1_token_id is None:
                    continue
                

                layer2_cond_accum = np.zeros(self.codebook_size, dtype=np.float64)
                total_samples = 0
                
                for batch_inputs in all_base_inputs:
                    probs = self._get_layer_probs(
                        batch_inputs, [c0_token_id, c1_token_id], layer=2
                    )
                    layer2_cond_accum += probs * len(batch_inputs)
                    total_samples += len(batch_inputs)
                
                layer2_cond_probs = layer2_cond_accum / total_samples
                

                for c2_idx in range(self.codebook_size):
                    if layer2_cond_probs[c2_idx] <= 0:
                        continue
                    c2_cond_prob = layer2_cond_probs[c2_idx]
                    joint_prob = float(joint_c0_c1_prob * c2_cond_prob)
                    if joint_prob > 1e-12:
                        all_joint_c0_c1_c2.append((c0_idx, c1_idx, c2_idx, joint_prob))
        
        print(f"  Found {len(all_joint_c0_c1_c2)} (C0, C1, C2) triplets")
        

        all_joint_c0_c1_c2.sort(key=lambda x: x[3], reverse=True)
        joint_probs_array = np.array([x[3] for x in all_joint_c0_c1_c2])
        cumsum = np.cumsum(joint_probs_array)
        total_prob = cumsum[-1] if len(cumsum) > 0 else 0
        
        if total_prob > 0:
            cumsum_normalized = cumsum / total_prob
        else:
            cumsum_normalized = cumsum
        
        cutoff_idx = np.searchsorted(cumsum_normalized, self.top_p) + 1
        cutoff_idx = min(cutoff_idx, len(all_joint_c0_c1_c2))
        
        kept_c0_c1_c2 = all_joint_c0_c1_c2[:cutoff_idx]
        kept_total = sum(x[3] for x in kept_c0_c1_c2)
        

        joint_c1_c2_c3: Dict[str, float] = {}
        layer2_marginal = np.zeros(self.codebook_size, dtype=np.float64)
        
        for c0_idx, c1_idx, c2_idx, joint_prob in kept_c0_c1_c2:
            normalized_prob = joint_prob / kept_total
            path_key = f"{c0_idx}_{c1_idx}_{c2_idx}"
            joint_c1_c2_c3[path_key] = normalized_prob
            layer2_marginal[c2_idx] += normalized_prob
        

        valid_probs = layer2_marginal[layer2_marginal > 0]
        layer2_entropy = -np.sum(valid_probs * np.log2(valid_probs))
        
        print(f"✓ Layer 2: top-{self.top_p*100:.0f}% kept {len(kept_c0_c1_c2)} paths")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [GPU memory] Layer 2 end: {mem_allocated:.2f} GB")
        

        max_entropy = float(np.log2(self.codebook_size))
        

        layer0_dist = {
            str(int(idx)): float(prob) 
            for idx, prob in zip(layer0_kept_indices, layer0_kept_probs)
        }
        

        layer1_dist = {
            str(idx): float(prob) 
            for idx, prob in enumerate(layer1_marginal) if prob > 1e-10
        }
        

        layer2_dist = {
            str(idx): float(prob) 
            for idx, prob in enumerate(layer2_marginal) if prob > 1e-10
        }
        
        return {
            "cycle": self.cycle,
            "num_users": len(inference_data),
            "codebook_size": self.codebook_size,
            "top_p": self.top_p,
            
            # Layer 0
            "layer0_num_tokens": len(layer0_kept_indices),
            "layer0_entropy": float(layer0_entropy),
            "layer0_normalized_entropy": float(layer0_entropy / max_entropy),
            "layer0_prob_distribution": layer0_dist,
            
            # Layer 1
            "layer1_num_paths": total_paths_c1_c2,
            "layer1_entropy": float(layer1_entropy),
            "layer1_normalized_entropy": float(layer1_entropy / max_entropy),
            "layer1_prob_distribution": layer1_dist,
            "layer1_joint_c1_c2": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in joint_c1_c2.items()},
            
            # Layer 2
            "layer2_num_paths": len(joint_c1_c2_c3),
            "layer2_entropy": float(layer2_entropy),
            "layer2_normalized_entropy": float(layer2_entropy / max_entropy),
            "layer2_prob_distribution": layer2_dist,
        }


def load_user_sequences(
    data_path: str, start_user: int = 1, end_user: int = None
) -> Dict[int, List[int]]:
    """Load user sequence data"""
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
            except ValueError:
                continue
    return user_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="OneRec multi-layer semantic code entropy calculation script (Top-p optimized)")


    parser.add_argument("--model_path", type=str, required=True, help="Trained model directory path")
    parser.add_argument("--data_path", type=str, required=True, help="User sequence data file path")
    parser.add_argument("--index_file", type=str, required=True, help="index.json file path")
    parser.add_argument("--output_path", type=str, required=True, help="Entropy stats output path")

    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p cumulative probability threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--codebook_size", type=int, default=256, help="RQ-VAE codebook size")

    parser.add_argument("--cycle", type=int, default=1, help="Current cycle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80)
    print("📊 OneRec multi-layer semantic code entropy calculation (Top-p optimized)")
    print("=" * 80)
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Index file: {args.index_file}")
    print(f"  - Output path: {args.output_path}")
    print("=" * 80 + "\n")

    print("Loading user data...")
    user_sequences = load_user_sequences(args.data_path)
    print(f"✓ Loaded {len(user_sequences)} users data\n")

    if not user_sequences:
        print("❌ Error: No user sequence data found")
        return

    calculator = OneRecMultilayerEntropyCalculator(args)
    results = calculator.calculate_multilayer_entropy(user_sequences)

    ensure_directory(os.path.dirname(args.output_path))
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"\n✓ Result saved to: {args.output_path}")

    print("\n" + "=" * 80)
    print("📈 Entropy stats results")
    print("=" * 80)
    print(f"  Layer 0: entropy={results['layer0_entropy']:.4f}, normalized entropy={results['layer0_normalized_entropy']:.4f}, tokens={results['layer0_num_tokens']}")
    print(f"  Layer 1: entropy={results['layer1_entropy']:.4f}, normalized entropy={results['layer1_normalized_entropy']:.4f}, paths={results['layer1_num_paths']}")
    print(f"  Layer 2: entropy={results['layer2_entropy']:.4f}, normalized entropy={results['layer2_normalized_entropy']:.4f}, paths={results['layer2_num_paths']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
