#!/usr/bin/env python3
"""
OneRec Recommendation Generator for Simulation
Generates top-k recommendations and converts SIDs to item_ids
"""

import pandas as pd
import fire
import torch
import json
import os
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList
)
from LogitProcessor import ConstrainedLogitsProcessor
import random
import numpy as np
from typing import Dict, List, Set, Any
from tqdm import tqdm
import copy
import logging
from datetime import datetime


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Tokenizer:
    """Tokenizer wrapper class"""
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


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_txt_data(txt_file: str) -> Dict[int, List[int]]:
    """
    Load txt format data
    
    Args:
        txt_file: txt file path
        
    Returns:
        {user_id: [item_ids]}
    """
    user_data = {}
    
    if not os.path.exists(txt_file):
        return user_data
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            try:
                user_id = int(parts[0])
                item_ids = [int(item) for item in parts[1:] if item.isdigit()]
                if item_ids:
                    user_data[user_id] = item_ids
            except ValueError:
                continue
    
    return user_data


def load_item_mappings(index_file: str, item_meta_path: str) -> Dict[str, Any]:
    """
    Load item_id to SID and title mappings
    
    Args:
        index_file: index.json file path
        item_meta_path: item metadata path
        
    Returns:
        Dictionary containing mapping information
    """
    mappings = {}
    
    # index_file = os.path.join(onerec_data_dir, f'{dataset_name}.index.json')
    with open(index_file, 'r') as f:
        item_to_sid = json.load(f)

        mappings['item_to_sid'] = {
            item_id: ''.join(tokens)
            for item_id, tokens in item_to_sid.items()
        }
    
    # item_meta_path = os.path.join(onerec_data_dir, f'{dataset_name}.item.json')
    with open(item_meta_path, 'r') as f:
        mappings['item_to_meta'] = json.load(f)
    
 print(f"✓ Loaded {len(mappings['item_to_sid'])} item-to-SID mappings")
 print(f"✓ Loaded {len(mappings['item_to_meta'])} item metadata")
    
    return mappings


def load_sid_to_itemid_mapping(info_file: str) -> Dict[str, str]:
    """
    Load SID to item_id mapping from info file
    
    Format: semantic_id \t item_title \t item_id
    """
    sid_to_itemid = {}
    
    with open(info_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                sid = parts[0].strip()
                item_id = parts[2].strip()
                sid_to_itemid[sid] = item_id
    
 print(f"✓ Loaded {len(sid_to_itemid)} SID to item_id mappings")
    return sid_to_itemid


def load_recommendation_history(history_file: str) -> Dict[str, Set[str]]:
    """
    Load recommendation history
    
    Format: user_id item1 item2 item3...
    """
    history = {}
    
    if not os.path.exists(history_file):
 print("⚠ Recommendation history file does not exist, skipping history-based filtering")
        return history
    
    with open(history_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = parts[0]
                items = set(parts[1:])
                history[user_id] = items
    
 print(f"✓ Loaded recommendation history for {len(history)} users")
    return history


def get_history(history_sids: List[str]) -> Dict[str, str]:
    """Get history information"""
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
    """Generate prompt"""
    return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


def main(
    model_path: str = "",
    data_path: str = "",
    info_file: str = "",
    index_file: str = "",
    item_meta_path: str = "",
    output_path: str = "",
    recommendation_history_file: str = "",
    k: int = 5,
    cycle: int = 1,
    end_user: int = None,
    batch_size: int = 1,
    seed: int = 42,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    category: str = "Toys_and_Games",
    exclude_recommended: bool = False,
    debug_log: str = None):
    """
    OneRec Recommendation Generator
    
    Args:
        model_path: Model checkpoint path
        data_path: Input data path (txt or CSV file, txt recommended)
        info_file: Info file path (SID to item_id mapping)
        index_file: index.json file path (item_id to SID mapping)
        item_meta_path: Item metadata path
        output_path: Output txt file path
        recommendation_history_file: Recommendation history file
        k: Number of recommended items
        cycle: Current cycle
        end_user: End user ID (generate recommendations for first end_user users)
        batch_size: Batch size
        seed: Random seed
        max_new_tokens: Maximum generated tokens
        num_beams: Beam search count
        category: Dataset category
        exclude_recommended: Whether to exclude already recommended items
        debug_log: Debug log file path (optional)
    """
    set_seed(seed)
    
    logger = logging.getLogger('recommend_gen')
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    if debug_log:
        os.makedirs(os.path.dirname(debug_log), exist_ok=True)
        file_handler = logging.FileHandler(debug_log, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("=" * 60)
 logger.info("OneRec Recommendation Generator")
    logger.info("=" * 60)
    
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books"
    }
    category_name = category_dict.get(category, "items")
 logger.info(f"Category: {category_name}")
    if debug_log:
 logger.info(f"Debug log: {debug_log}")
    
 logger.info(f"\n1. Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_obj = Tokenizer(tokenizer)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    model.generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    )
    
 logger.info(f"\n2. Loading SID mapping: {info_file}")
    sid_to_itemid = load_sid_to_itemid_mapping(info_file)
    
    with open(index_file, 'r') as f:
        item_to_sid_tokens = json.load(f)
    itemid_to_sid = {
        item_id: ''.join(tokens)
        for item_id, tokens in item_to_sid_tokens.items()
    }
    
 logger.info(f"\n3. Loading recommendation history...")
    rec_history = load_recommendation_history(recommendation_history_file) if exclude_recommended else {}
    
 logger.info(f"\n4. Loading user data: {data_path}")
    
    is_txt_file = data_path.endswith('.txt')
    
    if is_txt_file:
 logger.info(" Reading user sequences from txt file...")
        user_sequences = load_txt_data(data_path)
        
        # onerec_data_dir = os.path.dirname(index_file)
        # dataset_name = os.path.basename(index_file).replace('.index.json', '')
        item_mappings = load_item_mappings(index_file, item_meta_path)
        
        item_to_sid = item_mappings['item_to_sid']
        item_to_meta = item_mappings['item_to_meta']
        
        inference_data = []
        for user_id, items in user_sequences.items():
            if end_user and user_id > end_user:
                continue
            
            history_item_ids = []
            history_item_sids = []
            history_item_titles = []
            
            for item_id in items:
                if str(item_id) in item_to_sid and str(item_id) in item_to_meta:
                    history_item_ids.append(item_id)
                    history_item_sids.append(item_to_sid[str(item_id)])
                    history_item_titles.append(item_to_meta[str(item_id)].get('title', f'Item_{item_id}'))
            
            if history_item_ids:
                inference_data.append({
                    'user_id': user_id,
                    'history_item_id': history_item_ids,
                    'history_item_sid': history_item_sids,
                    'history_item_title': history_item_titles
                })
        
 logger.info(f"✓ Loaded {len(inference_data)} users from txt")
        
    else:
 logger.info(" Reading from CSV file...")
        df = pd.read_csv(data_path)
        
        users_df = df.groupby('user_id').tail(1).reset_index(drop=True)
        
        if end_user:
            users_df['user_id_num'] = users_df['user_id'].str.replace('A', '').astype(int)
            users_df = users_df[users_df['user_id_num'] <= end_user].copy()
            users_df = users_df.drop('user_id_num', axis=1)
        
        inference_data = []
        for _, row in users_df.iterrows():
            user_id_str = row['user_id'].replace('A', '')
            user_id = int(user_id_str)
            
            if end_user and user_id > end_user:
                continue
            
            try:
                history_item_ids = eval(row['history_item_id']) if isinstance(row['history_item_id'], str) else row['history_item_id']
                history_item_sids = eval(row['history_item_sid']) if isinstance(row['history_item_sid'], str) else row['history_item_sid']
                history_item_titles = eval(row['history_item_title']) if isinstance(row['history_item_title'], str) else row['history_item_title']
            except:
                continue
            
            inference_data.append({
                'user_id': user_id,
                'history_item_id': history_item_ids,
                'history_item_sid': history_item_sids,
                'history_item_title': history_item_titles
            })
        
 logger.info(f"✓ Loaded {len(inference_data)} users from CSV")
    
 logger.info(f"✓ Prepared {len(inference_data)} users for recommendation generation")
    
 logger.info(f"\n5. Building constraint dictionary...")
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
    
    if model_path.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    
    if model_path.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])
    
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    
 logger.info(f"✓ Built constraint dictionary with {len(hash_dict)} entries")
 logger.debug(f"Constraint dict sample (5): {list(hash_dict.items())[:5]}")
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
    
 logger.info(f"\n6. Generating recommendations...")
    
    all_recommendations = {}
    
    num_batches = (len(inference_data) + batch_size - 1) // batch_size
    
 for batch_idx in tqdm(range(num_batches), desc="Generating recommendations"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(inference_data))
        batch_users = inference_data[batch_start:batch_end]
        
        encodings = []
        user_ids = []
        
        for user_data in batch_users:
            user_id = str(user_data['user_id'])
            user_ids.append(user_id)
            
            history_sids = user_data.get('history_item_sid', [])
            if not isinstance(history_sids, list):
                history_sids = []
            
            instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
            tokens = tokenizer_obj.encode(instruction, bos=True, eos=False)
            
            history = get_history(history_sids)
            prompt = generate_prompt(history)
            tokens = tokens + tokenizer_obj.encode(prompt, bos=False, eos=False)
            
            attention_mask = [1] * len(tokens)
            
            encodings.append({
                "input_ids": tokens,
                "attention_mask": attention_mask,
            })
        
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        
        padding_encodings = {"input_ids": []}
        attention_mask = []
        
        for _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            do_sample = False,
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=model_path,
                eos_token_id=tokenizer.eos_token_id,
                prompt_len=maxLen
            )
            logits_processor = LogitsProcessorList([clp])
            
            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        if model_path.lower().find("llama") > -1:
            raw_output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            raw_output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
        
        if batch_idx == 0:
 logger.debug(f"Raw generation samples (first 3):")
            for idx, raw in enumerate(raw_output[:3]):
                logger.debug(f"  [{idx}] {raw[:200]}")
        
        output = [_.split("Response:\n")[-1].strip() for _ in raw_output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        
        stats = clp.get_stats()
        if batch_idx % 100 == 0:
 logger.info(f"[batch {batch_idx}] Stats - Total: {stats['total_count']}, Empty: {stats['empty_count']}, Ratio: {stats['empty_ratio']:.2%}")
        
        for i, user_id in enumerate(user_ids):
            user_outputs = real_outputs[i]
            
            logger.debug(f"\n{'='*60}")
 logger.debug(f"User {user_id} generated SIDs (first 10):")
            for idx, sid in enumerate(user_outputs[:10]):
                logger.debug(f"  {idx+1}. '{sid}'")
            
            recommended_items = []
            seen_items = set()
            
            user_history = rec_history.get(user_id, set())
            
 logger.debug(f"User {user_id} has {len(user_history)} items in recommendation history")
            
            sid_to_item_success = 0
            sid_to_item_fail = 0
            duplicate_items = 0
            history_filtered = 0
            
            for sid in user_outputs:
                if len(recommended_items) >= k:
                    break
                
                item_id = sid_to_itemid.get(sid)
                
                if not item_id:
                    sid_to_item_fail += 1
 logger.debug(f" SID mapping failed: '{sid}'")
                    continue
                
                sid_to_item_success += 1
                
                if item_id in seen_items:
                    duplicate_items += 1
 logger.debug(f" Duplicate item: {item_id} (SID: {sid})")
                    continue
                
                if exclude_recommended and item_id in user_history:
                    history_filtered += 1
 logger.debug(f" Filtered by history: {item_id} (SID: {sid})")
                    continue
                
                seen_items.add(item_id)
                recommended_items.append(item_id)
 logger.debug(f" ✓ Added to recommendations: {item_id} (SID: {sid})")
            
 logger.debug(f"\nUser {user_id} recommendation stats:")
            logger.debug(f"  - Total beam outputs: {len(user_outputs)}")
 logger.debug(f" - SID mapping success: {sid_to_item_success}")
 logger.debug(f" - SID mapping failed: {sid_to_item_fail}")
 logger.debug(f" - Duplicate items: {duplicate_items}")
 logger.debug(f" - Filtered by history: {history_filtered}")
 logger.debug(f" - Final recommendations: {len(recommended_items)}")
            
            if len(recommended_items) > 0:
 logger.debug(f" Recommended items: {recommended_items}")
            
            if len(recommended_items) < k:
 logger.warning(f"⚠ User {user_id} generated only {len(recommended_items)} valid recommendations (target: {k})")
                
                if sid_to_item_fail > 0:
                    failed_sids = [sid for sid in user_outputs[:20] if sid not in sid_to_itemid]
                    if failed_sids:
 logger.warning(f" Failed SID samples (first 5): {failed_sids[:5]}")
            
            all_recommendations[user_id] = recommended_items
    
 logger.info(f"\n7. Saving recommendation results: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for user_id in sorted(all_recommendations.keys(), key=int):
            items = all_recommendations[user_id]
            if items:
                f.write(f"{user_id} {' '.join(items)}\n")
    
 logger.info(f"✓ Generated recommendations for {len(all_recommendations)} users")
    logger.info("=" * 60)
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
 logger.info("GPU memory cleared")


if __name__ == '__main__':
    fire.Fire(main)

