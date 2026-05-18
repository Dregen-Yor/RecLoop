# -*- coding: utf-8 -*-
"""
Preprocess the full Toys_and_Games.jsonl.gz dataset (Amazon 2023) to create 
a txt file compatible with run_finetune_full.py

This script:
1. Loads the gzipped JSONL data
2. Applies k-core filtering
3. Sorts interactions by timestamp
4. Removes consecutive duplicate items
5. Saves in the format expected by get_user_seqs(): "user_id item1 item2 item3..."
"""

import os
import gzip
import json
import argparse
import numpy as np
from collections import defaultdict


def load_and_process_jsonl_gz(jsonl_gz_path, min_user_interactions=5, min_item_interactions=5):
    """
    Load the gzipped JSONL data and apply k-core filtering.
    """
    print(f"Loading data from {jsonl_gz_path}...")
    
    # Load raw data
    user_items = defaultdict(list)  # user_id -> [(timestamp, asin), ...]
    line_count = 0
    
    with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f" {line_count:,} ...")
            
            try:
                record = json.loads(line.strip())
                user_id = record.get('user_id')
                asin = record.get('asin') or record.get('parent_asin')
                timestamp = record.get('timestamp', 0)
                
                if user_id and asin:
                    user_items[user_id].append((timestamp, asin))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(user_items):,} users with {sum(len(v) for v in user_items.values()):,} interactions")
    
    # Apply k-core filtering iteratively
    print(f"Applying {min_user_interactions}/{min_item_interactions}-core filtering...")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Count item frequencies
        item_count = defaultdict(int)
        for user_id, items in user_items.items():
            for _, asin in items:
                item_count[asin] += 1
        
        # Filter items that appear less than min_item_interactions times
        valid_items = {asin for asin, count in item_count.items() 
                      if count >= min_item_interactions}
        
        # Filter user sequences
        new_user_items = {}
        for user_id, items in user_items.items():
            filtered = [(ts, asin) for ts, asin in items if asin in valid_items]
            if len(filtered) >= min_user_interactions:
                new_user_items[user_id] = filtered
        
        print(f"  Iteration {iteration}: Users: {len(new_user_items):,}, Valid items: {len(valid_items):,}")
        
        # Check if converged
        if len(new_user_items) == len(user_items):
            break
        
        user_items = new_user_items
        
        # Safety check to prevent infinite loop
        if iteration > 100:
            print("Warning: k-core filtering did not converge after 100 iterations")
            break
    
    print(f"After filtering: {len(user_items):,} users, {len(valid_items):,} items")
    
    # Sort by timestamp and remove consecutive duplicates
    print("Sorting by timestamp and removing consecutive duplicates...")
    for user_id in user_items:
        items = user_items[user_id]
        # Sort by timestamp
        items.sort(key=lambda x: x[0])
        # Remove consecutive duplicates
        deduped = []
        for ts, asin in items:
            if not deduped or deduped[-1][1] != asin:
                deduped.append((ts, asin))
        user_items[user_id] = deduped
    
    # Filter again after deduplication
    user_items = {uid: items for uid, items in user_items.items() 
                  if len(items) >= min_user_interactions}
    
    print(f"After deduplication: {len(user_items):,} users")
    
    # Create mappings
    print("Creating ID mappings...")
    
    # Collect all items and create mapping
    all_items = set()
    for items in user_items.values():
        for _, asin in items:
            all_items.add(asin)
    
    # Item mapping: 0 is reserved for padding, so start from 1
    item_mapping = {asin: idx + 1 for idx, asin in enumerate(sorted(all_items))}
    
    # User mapping: 0-indexed for array access
    user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(user_items.keys()))}
    
    # Create user sequences with mapped IDs
    user_seq = []
    for user_id in sorted(user_items.keys()):
        items = user_items[user_id]
        mapped_items = [item_mapping[asin] for _, asin in items]
        user_seq.append(mapped_items)
    
    max_item = len(item_mapping)
    total_interactions = sum(len(s) for s in user_seq)
    
    print(f"\n=== Final Dataset Statistics ===")
    print(f"Number of users: {len(user_seq):,}")
    print(f"Number of items: {max_item:,}")
    print(f"Total interactions: {total_interactions:,}")
    print(f"Average sequence length: {total_interactions / len(user_seq):.2f}")
    print(f"Sparsity: {1 - total_interactions / (len(user_seq) * max_item):.6f}")
    
    return user_seq, max_item, user_mapping, item_mapping


def save_processed_data(user_seq, output_path):
    """Save processed data in the expected txt format for run_finetune_full.py."""
    print(f"Saving processed data to {output_path}...")
    with open(output_path, 'w') as f:
        for user_id, items in enumerate(user_seq):
            items_str = ' '.join(map(str, items))
            f.write(f"{user_id} {items_str}\n")
    print(f"Saved {len(user_seq):,} user sequences")


def main():
    parser = argparse.ArgumentParser(description="Preprocess full Amazon 2023 Toys_and_Games dataset")
    
    parser.add_argument('--input_file', 
                       default='./data/Toys_and_Games.jsonl.gz',
                       type=str, help="Path to input .jsonl.gz file")
    parser.add_argument('--output_dir', 
                       default='./data/Toys_and_Games_Full/',
                       type=str, help="Output directory for processed data")
    parser.add_argument('--output_name', 
                       default='Toys_and_Games_Full',
                       type=str, help="Name for output files (without extension)")
    parser.add_argument('--min_user_interactions', type=int, default=5,
                       help="Minimum interactions per user for k-core filtering")
    parser.add_argument('--min_item_interactions', type=int, default=5,
                       help="Minimum interactions per item for k-core filtering")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process data
    user_seq, max_item, user_mapping, item_mapping = load_and_process_jsonl_gz(
        args.input_file,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions
    )
    
    # Save processed data
    output_txt = os.path.join(args.output_dir, f'{args.output_name}.txt')
    save_processed_data(user_seq, output_txt)
    
    # Save mappings for reference
    np.save(os.path.join(args.output_dir, f'{args.output_name}_user_mapping.npy'), 
            user_mapping, allow_pickle=True)
    np.save(os.path.join(args.output_dir, f'{args.output_name}_item_mapping.npy'), 
            item_mapping, allow_pickle=True)
    
    print(f"\nProcessing complete!")
    print(f"Output files saved to: {args.output_dir}")
    print(f"\nTo train SASRec on this data, run:")
    print(f"  python run_finetune_full.py --data_dir ./data/ --data_name {args.output_name} --backbone SASRec")


if __name__ == '__main__':
    main()
