#!/usr/bin/env python3
"""
Simulation Data Converter for OneRec
Converts simulation txt format to OneRec CSV format with full augmentation
"""

import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional
import argparse


def load_item_mappings(onerec_data_dir: str, dataset_name: str, Text_mode: int = 1) -> Dict[str, Any]:
    """
    Load item_id to SID and title mappings
    
    Args:
        onerec_data_dir: OneRec data directory
        dataset_name: Dataset name
        Text_mode: Whether in semantic mode
        
    Returns:
        Dictionary containing mapping information
    """
    mappings = {}
    
    print("Text_mode: ", Text_mode)
    if Text_mode == 1:
        index_file = os.path.join(onerec_data_dir, f'{dataset_name}.index.json')
    else:
        index_file = os.path.join(onerec_data_dir, f'{dataset_name}-sasrec.index.json')
    with open(index_file, 'r') as f:
        item_to_sid = json.load(f)

        mappings['item_to_sid'] = {
            item_id: ''.join(tokens)
            for item_id, tokens in item_to_sid.items()
        }
    
    item_file = os.path.join(onerec_data_dir, f'{dataset_name}.item.json')
    with open(item_file, 'r') as f:
        mappings['item_to_meta'] = json.load(f)
    
 print(f"✓ Loaded {len(mappings['item_to_sid'])} item-to-SID mappings")
 print(f"✓ Loaded {len(mappings['item_to_meta'])} item metadata")
    
    return mappings


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


def _build_csv_row(user_id: int, history: List[int], target_item: int,
                   item_to_sid: Dict[str, str], item_to_meta: Dict[str, Any],
                   max_history_length: int = 10) -> Optional[Dict[str, Any]]:
    """
    Build CSV row (helper function)
    
    Args:
        user_id: User ID
        history: History item list
        target_item: Target item
        item_to_sid: item_id to SID mapping
        item_to_meta: item_id to metadata mapping
        max_history_length: Maximum history length (sliding window)
        
    Returns:
        CSV row dictionary, returns None if target item not in mappings
    """

    if str(target_item) not in item_to_sid or str(target_item) not in item_to_meta:
        return None
    
    st = max(len(history) - max_history_length, 0)
    history_window = history[st:]
    
    history_item_ids = []
    history_item_sids = []
    history_item_titles = []
    
    for h_item in history_window:
        if str(h_item) in item_to_sid and str(h_item) in item_to_meta:
            history_item_ids.append(h_item)
            history_item_sids.append(item_to_sid[str(h_item)])
            history_item_titles.append(item_to_meta[str(h_item)].get('title', f'Item_{h_item}'))
    
    target_sid = item_to_sid[str(target_item)]
    target_title = item_to_meta[str(target_item)].get('title', f'Item_{target_item}')
    
    row = {
        'user_id': f'A{user_id}',
        'history_item_title': history_item_titles,
        'item_title': target_title,
        'history_item_id': history_item_ids,
        'item_id': target_item,
        'history_item_sid': history_item_sids,
        'item_sid': target_sid
    }
    
    return row


def convert_txt_to_csv_full_augmentation(
    current_txt: str,
    mappings: Dict[str, Any],
    output_train_csv: str,
    output_valid_csv: str,
    output_test_csv: str,
    dataset_name: str,
    max_history_length: int = 10
) -> Dict[str, Any]:
    """
    Full augmentation and implement Leave One Out split
    
    Generate all subsequences for each user's complete sequence and split into:
    - Training set: Augmented samples from first N-2 items
    - Validation set: Second to last item
    - Test set: Last item
    
    For sequence [1,2,3,4,5]:
    - Training: [1]→2, [1,2]→3
    - Validation: [1,2,3]→4
    - Test: [1,2,3,4]→5
    
    Args:
        current_txt: Current cycle txt file
        mappings: Item mapping information
        output_train_csv: Output training CSV file
        output_valid_csv: Output validation CSV file
        output_test_csv: Output test CSV file
        dataset_name: Dataset name
        max_history_length: Maximum history length (sliding window size)
        
    Returns:
        Conversion statistics
    """
    item_to_sid = mappings['item_to_sid']
    item_to_meta = mappings['item_to_meta']
    
    current_data = load_txt_data(current_txt)
    
 print(f"Current data: {len(current_data)} users")
    
    train_interactions = []
    valid_interactions = []
    test_interactions = []
    skipped_items = 0
    skipped_users = 0
    
    for user_id, items in current_data.items():
        if len(items) < 3:
            skipped_users += 1
            continue
        
        for i in range(1, len(items) - 2):
            history = items[:i]
            target_item = items[i]
            
            row = _build_csv_row(user_id, history, target_item, item_to_sid, item_to_meta, max_history_length)
            if row:
                train_interactions.append(row)
            else:
                skipped_items += 1
        
        valid_idx = len(items) - 2
        history = items[:valid_idx]
        target_item = items[valid_idx]
        
        row = _build_csv_row(user_id, history, target_item, item_to_sid, item_to_meta, max_history_length)
        if row:
            valid_interactions.append(row)
        else:
            skipped_items += 1
        
        test_idx = len(items) - 1
        history = items[:test_idx]
        target_item = items[test_idx]
        
        row = _build_csv_row(user_id, history, target_item, item_to_sid, item_to_meta, max_history_length)
        if row:
            test_interactions.append(row)
        else:
            skipped_items += 1
    
    stats = {
        'total_users': len(current_data),
        'train_samples': len(train_interactions),
        'valid_samples': len(valid_interactions),
        'test_samples': len(test_interactions),
        'skipped_items': skipped_items,
        'skipped_users': skipped_users,
        'output_train': output_train_csv,
        'output_valid': output_valid_csv,
        'output_test': output_test_csv
    }
    
    if train_interactions:
        df_train = pd.DataFrame(train_interactions)
        os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
        df_train.to_csv(output_train_csv, index=False)
 print(f"✓ Training: {len(train_interactions)} samples → {output_train_csv}")
    else:
 print(f"⚠ No training samples generated")
    
    if valid_interactions:
        df_valid = pd.DataFrame(valid_interactions)
        os.makedirs(os.path.dirname(output_valid_csv), exist_ok=True)
        df_valid.to_csv(output_valid_csv, index=False)
 print(f"✓ Validation: {len(valid_interactions)} samples → {output_valid_csv}")
    else:
 print(f"⚠ No validation samples generated")
    
    if test_interactions:
        df_test = pd.DataFrame(test_interactions)
        os.makedirs(os.path.dirname(output_test_csv), exist_ok=True)
        df_test.to_csv(output_test_csv, index=False)
 print(f"✓ Test: {len(test_interactions)} samples → {output_test_csv}")
    else:
 print(f"⚠ No test samples generated")
    
    if train_interactions:
 print(f"\nTraining sample:")
        print(f"  user_id: {train_interactions[0]['user_id']}")
        print(f"  history_item_id: {train_interactions[0]['history_item_id'][:3]}...")
        print(f"  item_id: {train_interactions[0]['item_id']}")
    
    if skipped_items > 0:
 print(f"⚠ {skipped_items} items skipped (missing mappings)")
    if skipped_users > 0:
 print(f"⚠ {skipped_users} users skipped (sequence length < 3)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert simulation txt data to OneRec CSV format (Leave One Out)'
    )
    parser.add_argument('--current_txt', type=str, required=True,
 help='Current cycle txt data file')
    parser.add_argument('--onerec_data_dir', type=str, required=True,
 help='OneRec data directory (contains index.json and item.json)')
    parser.add_argument('--output_train_csv', type=str, required=True,
 help='Output training CSV file path')
    parser.add_argument('--output_valid_csv', type=str, required=True,
 help='Output validation CSV file path')
    parser.add_argument('--output_test_csv', type=str, required=True,
 help='Output test CSV file path')
    parser.add_argument('--dataset_name', type=str, required=True,
 help='Dataset name')
    parser.add_argument('--max_history_length', type=int, default=10,
 help='Maximum history length (sliding window, default 10)')
    parser.add_argument('--Text_mode', type=int, default=0,
 help='SID generation mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
 print("OneRec data conversion (Leave One Out)")
    print("=" * 60)
    
 print("\n1. Loading item mappings...")
    mappings = load_item_mappings(args.onerec_data_dir, args.dataset_name, args.Text_mode)
    
 print("\n2. Performing Leave One Out data split...")
    stats = convert_txt_to_csv_full_augmentation(
        args.current_txt,
        mappings,
        args.output_train_csv,
        args.output_valid_csv,
        args.output_test_csv,
        args.dataset_name,
        max_history_length=args.max_history_length
    )
    
 print("\n3. Conversion completed!")
 print(f" Total users: {stats['total_users']}")
 print(f" Training samples: {stats['train_samples']}")
 print(f" Validation samples: {stats['valid_samples']}")
 print(f" Test samples: {stats['test_samples']}")
 print(f" Training file: {stats['output_train']}")
 print(f" Validation file: {stats['output_valid']}")
 print(f" Test file: {stats['output_test']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

