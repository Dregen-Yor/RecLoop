import argparse
import json
import gzip
import os
import ast
from typing import Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


def setup_logging(args):
    """Configure logging system"""
    log_filename = f"./{args.dataset}/data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized, log file: {log_filename}")
    return logger


def parse_jsonl_gz(path):
    """Parse compressed JSONL file, supports standard JSON and Python dict format (single quotes)"""
    logger.info(f"Starting to parse file: {path}")
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(line)
                        yield data
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Line {line_num} parsing failed, error: {e}")
                        continue
    except Exception as e:
        logger.error(f"Failed to parse compressed file: {path}, error: {e}")
        raise


def parse_jsonl(path):
    """Parse uncompressed JSONL file, supports standard JSON and Python dict format (single quotes)"""
    logger.info(f"Starting to parse file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(line)
                        yield data
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Line {line_num} parsing failed, error: {e}")
                        continue
    except Exception as e:
        logger.error(f"Failed to parse file: {path}, error: {e}")
        raise


def process_reviews():
    """Process review data and save as JSON format"""
    logger.info("Starting to process reviews...")

    output_file = f"./{args.dataset}/{args.dataset}.json"
    item_set = set()
    review_count = 0
    if not os.path.exists(output_file):

        if os.path.exists(f"./Amazon2014/reviews_{args.dataset}_5.json.gz"):
            logger.info(f"Found compressed file: ./Amazon2014/reviews_{args.dataset}_5.json.gz")
            reviews = parse_jsonl_gz(f"./Amazon2014/reviews_{args.dataset}_5.json.gz")
        logger.info(f"Starting to write review data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for review in reviews:
                f.write(json.dumps(review, ensure_ascii=False) + '\n')
                review_count += 1
                
                # item_id = review.get('parent_asin') or review.get('asin')
                item_id = review.get('asin')
                if item_id:
                    item_set.add(item_id)
                
                if review_count % 10000 == 0:
                    logger.info(f"Processed {review_count} reviews")

    end_time = datetime.now()
    
    logger.info(f"Review data processing completed")
    logger.info(f"Processed review count: {review_count}")
    logger.info(f"Original item count: {len(item_set)}")

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                first_review = json.loads(first_line)
                logger.info(f"First review example: {first_review}")
    except Exception as e:
        logger.warning(f"Failed to read first review: {e}")


def process_metadata(args):
    """Process item metadata"""
    logger.info("Starting to process metadata...")
    
    output_file = f"./{args.dataset}/{args.dataset}_metadata.json"
    if os.path.exists(output_file):
        return

    if os.path.exists(f"./Amazon2014/meta_{args.dataset}.jsonl.gz"):
        logger.info(f"Found compressed metadata file: ./Amazon2014/meta_{args.dataset}.jsonl.gz")
        metadata_list = parse_jsonl_gz(f"./Amazon2014/meta_{args.dataset}.jsonl.gz")
    elif os.path.exists(f"./Amazon2014/meta_{args.dataset}.json.gz"):
        logger.info(f"Found compressed metadata file: ./Amazon2014/meta_{args.dataset}.json.gz")
        metadata_list = parse_jsonl_gz(f"./Amazon2014/meta_{args.dataset}.json.gz")
    else:
        logger.error(f"Metadata file not found: ./Amazon2014/meta_{args.dataset}.jsonl.gz or ./Amazon2014/meta_{args.dataset}.json.gz")
        return
    
    metadata_count = 0
    logger.info(f"Starting to write metadata to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for metadata in metadata_list:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                metadata_count += 1
                
                if metadata_count % 1000 == 0:
                    logger.info(f"Processed {metadata_count} metadata entries")
        
        logger.info(f"Metadata processing completed")
        logger.info(f"Metadata count: {metadata_count}")
        
    except Exception as e:
        logger.error(f"Failed to process metadata: {e}")
        raise


def remove_consecutive_duplicates(user_items):
    """Remove consecutive duplicate items in each user sequence, keep only one"""
    logger.info("Starting to remove consecutive duplicate items...")
    start_time = datetime.now()
    
    total_before = sum(len(items) for items in user_items.values())
    removed_count = 0
    
    for user_id in user_items:
        original_items = user_items[user_id]
        if not original_items:
            continue
        
        deduplicated_items = []
        prev_item = None
        for item in original_items:
            if item != prev_item:
                deduplicated_items.append(item)
                prev_item = item
            else:
                removed_count += 1
        
        user_items[user_id] = deduplicated_items
    
    total_after = sum(len(items) for items in user_items.values())
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Consecutive duplicate removal completed")
    logger.info(f"Interactions before deduplication: {total_before}")
    logger.info(f"Interactions after deduplication: {total_after}")
    logger.info(f"Removed consecutive duplicates: {removed_count}")
    logger.info(f"Processing time: {processing_time:.2f} s")
    with open(f'./{args.dataset}/{args.dataset}-before.txt', 'w', encoding='utf-8') as f:
        for user, interactions in user_items.items():
            if len(interactions) >= 3:
                # users_with_sufficient_interactions += 1
                
                # train_items = [i['item'] for i in interactions[:-2]]
                # items = [i for i in interactions]
                # val_item = interactions[-2]['item']
                # test_item = interactions[-1]['item']
                # all_data.append(user, items)
                f.write(f'{user} {" ".join(str(item) for item in interactions)}\n')
    return user_items


def create_mappings(args):
    """Create user ID and item ID integer mappings"""
    logger.info("Starting to create user and item mappings...")
    start_time = datetime.now()

    user_items = defaultdict(list)
    logger.info("Collecting user-item interactions...")

    try:
        with open(f"./{args.dataset}/{args.dataset}.json", 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                review = json.loads(line.strip())
                user_id = review.get('reviewerID')
                
                item_id = review.get('asin')
                
                if user_id and item_id:
                    user_items[user_id].append(item_id)
                
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count} reviews for mapping")

        logger.info(f"Stats: users={len(user_items)}, total interactions={sum(len(items) for items in user_items.values())}")

        logger.info("Starting to create user and item mappings...")
        user_mapping = {}
        item_mapping = {}
        item_set = set()

        for user_id, items in user_items.items():
            if user_id not in user_mapping:
                user_mapping[user_id] = len(user_mapping) + 1
            
            for item_id in items:
                if item_id not in item_mapping:
                    item_mapping[item_id] = len(item_mapping) + 1
                item_set.add(item_id)

        logger.info("Mapping creation completed (final mapping will be saved after data split)")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        logger.info(f"Mapping creation completed")
        logger.info(f"User count: {len(user_mapping)}")
        logger.info(f"Item count: {len(item_mapping)}")
        logger.info(f"First 5 user mappings: {list(user_mapping.items())[:5]}")
        logger.info(f"First 5 item mappings: {list(item_mapping.items())[:5]}")
        logger.info(f"Mapping creation time: {processing_time:.2f} s")

        return user_mapping, item_mapping
        
    except Exception as e:
        logger.error(f"Failed to create mapping: {e}")
        raise


def remove_consecutive_duplicates_interactions(user_interactions):
    """Remove consecutive duplicate items in each user sequence, keep only one (for user_interactions format)"""
    logger.info("Starting to remove consecutive duplicate items...")
    start_time = datetime.now()
    
    total_before = sum(len(interactions) for interactions in user_interactions.values())
    removed_count = 0
    
    for user_id in user_interactions:
        original_interactions = user_interactions[user_id]
        if not original_interactions:
            continue
        
        deduplicated_interactions = []
        prev_item = None
        for interaction in original_interactions:
            current_item = interaction['item']
            if current_item != prev_item:
                deduplicated_interactions.append(interaction)
                prev_item = current_item
            else:
                removed_count += 1
        
        user_interactions[user_id] = deduplicated_interactions
    
    total_after = sum(len(interactions) for interactions in user_interactions.values())
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Consecutive duplicate removal completed")
    logger.info(f"Interactions before deduplication: {total_before}")
    logger.info(f"Interactions after deduplication: {total_after}")
    logger.info(f"Removed consecutive duplicates: {removed_count}")
    logger.info(f"Processing time: {processing_time:.2f} s")
    
    return user_interactions


def split_data(args):

    user_items = defaultdict(list)

    try:
        with open(f"./{args.dataset}/{args.dataset}.json", 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                review = json.loads(line.strip())
                user_id = review.get('reviewerID')
                # item_id = review.get('parent_asin') or review.get('asin')
                item_id = review.get('asin')
                timestamp = review.get('unixReviewTime')

                if user_id and item_id and timestamp:
                    user_items[user_id].append({
                        'item': item_id,
                        'timestamp': timestamp
                    })
                
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count} reviews")

        logger.info(f"Completed stats: users={len(user_items)}, total interactions={sum(len(items) for items in user_items.values())}")

        for user in user_items:
            user_items[user].sort(key=lambda x: x['timestamp'])
        
        user_item_lists = defaultdict(list)
        for user_id, interactions in user_items.items():
            user_item_lists[user_id] = [i['item'] for i in interactions]
        
        removed_count = 0
        for user_id in user_item_lists:
            original_items = user_item_lists[user_id]
            if not original_items:
                continue
            
            deduplicated_items = []
            prev_item = None
            for item in original_items:
                if item != prev_item:
                    deduplicated_items.append(item)
                    prev_item = item
                else:
                    removed_count += 1
            
            user_item_lists[user_id] = deduplicated_items
        
        valid_user_items = {}
        for user_id, items in user_item_lists.items():
            if len(items) >= 3:
                valid_user_items[user_id] = items
        
        logger.info(f"Users with sufficient interactions: {len(valid_user_items)}")
        
        logger.info("Starting to create user and item mappings...")
        
        valid_items = set()
        for items in valid_user_items.values():
            valid_items.update(items)
        
        logger.info(f"Valid users: {len(valid_user_items)}")
        logger.info(f"Valid items: {len(valid_items)}")
        
        sorted_users = sorted(valid_user_items.keys())
        sorted_items = sorted(valid_items)
        
        user_mapping = {user_id: idx for idx, user_id in enumerate(sorted_users, start=1)}
        item_mapping = {item_id: idx for idx, item_id in enumerate(sorted_items, start=1)}
        
        logger.info(f"User ID range: 1 ~ {len(user_mapping)}")
        logger.info(f"Item ID range: 1 ~ {len(item_mapping)}")
        
        logger.info("Starting to save mapping file...")
        np.save(f'./{args.dataset}/user_mapping.npy', user_mapping)
        np.save(f'./{args.dataset}/item_mapping.npy', item_mapping)
        
        logger.info(f"First 5 user mappings: {list(user_mapping.items())[:5]}")
        logger.info(f"First 5 item mappings: {list(item_mapping.items())[:5]}")
        
        logger.info("Starting to map ID data file...")
        users_written = 0
        total_interactions_written = 0
        
        with open(f'./{args.dataset}/{args.dataset}.txt', 'w', encoding='utf-8') as f:
            for user_id, items in valid_user_items.items():
                new_user_id = user_mapping[user_id]
                new_items = [item_mapping[item] for item in items]
                f.write(f'{new_user_id} {" ".join(str(item) for item in new_items)}\n')
                users_written += 1
                total_interactions_written += len(new_items)

        logger.info(f"Users written: {users_written}")
        logger.info(f"Total interactions written: {total_interactions_written}")
        
        logger.info(f"Data processing completed")
        
        return user_mapping, item_mapping
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise


def generate_item_embeddings(item_mapping):
    """Generate item semantic embeddings using SentenceTransformer"""
    logger.info("Starting to generate item embeddings...")
    start_time = datetime.now()
    
    item_info = {}
    logger.info("Starting to process data...")
    
    try:
        if os.path.exists(f"./{args.dataset}/{args.dataset}_metadata.json"):
            with open(f"./{args.dataset}/{args.dataset}_metadata.json", 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    metadata = json.loads(line.strip())
                    # asin = metadata.get('parent_asin')
                    asin = metadata.get('asin')
                    if asin and asin in item_mapping:
                        item_id = item_mapping[asin]
                        item_info[item_id] = {
                            'item_id': item_id,
                            'title': metadata.get('title', ''),
                            'description': metadata.get('description', ''),
                            'categories': metadata.get('categories', []),
                            'price': metadata.get('price', '')
                        }
                    
                    if line_count % 1000 == 0:
                        logger.info(f"Processed {line_count} metadata entries, found {len(item_info)} valid")
            
            logger.info(f"Data processing completed, found {len(item_info)} items data")
        else:
            logger.error(f"Metadata file does not exist: {args.dataset}_metadata.json")
            return
        
        logger.info("Loading SentenceTransformer model...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-t5-xl')
            logger.info("Model loaded from: sentence-t5-xl")
        except Exception as e:
            logger.info("Model not found locally, downloading...")
            model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
            logger.info("Model downloaded successfully")
        
        logger.info("Starting to generate embeddings...")
        item_embeddings = []
        
        batch_size = 32
        item_ids = []
        texts = []
        
        for item_id, info in item_info.items():
            categories_str = ", ".join(info['categories'][0]) if info['categories'] else ""
            text = f"Item ID: {item_id} | Title: {info['title']} | Description: {info['description']} | Categories: {categories_str} "
            item_ids.append(item_id)
            texts.append(text)
            print(text)
        
        logger.info(f"Starting to generate embeddings for {len(texts)} items, batch size: {batch_size}")
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_item_ids = item_ids[i:i+batch_size]
            
            batch_embeddings = model.encode(batch_texts)
            
            for item_id, embedding in zip(batch_item_ids, batch_embeddings):
                item_embeddings.append({
                    'Item_id': item_id,
                    'embedding': embedding.tolist()
                })
        
        logger.info("Starting to save embedding file...")
        emb_df = pd.DataFrame(item_embeddings)
        emb_df.to_parquet(f'./{args.dataset}/item_emb.parquet', index=False)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Embedding generation completed")
        logger.info(f"Total items embedded: {emb_df.shape}")
        logger.info(f"Embedding dimension: {len(emb_df.iloc[0]['embedding'])}")
        logger.info(f"Embedding generation time: {processing_time:.2f} s")
        logger.info(f"First 3 items: {emb_df.head(3)}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(f"./{args.dataset}", exist_ok=True)
    logger = setup_logging(args)
    logger.info("=" * 60)
    logger.info("Starting data processing")
    logger.info("=" * 60)

    total_start_time = datetime.now()
    
    try:
        logger.info("Step 1: Processing review data")
        process_reviews()
        
        logger.info("Step 2: Processing metadata")
        process_metadata(args)
        
        logger.info("Step 3: Data processing and splitting")
        user_mapping, item_mapping = split_data(args)

        total_end_time = datetime.now()
        total_processing_time = (total_end_time - total_start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("TIGER Data processing pipeline completed")
        logger.info(f"Total processing time: {total_processing_time:.2f} s ({total_processing_time/60:.2f} min)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        logger.error("=" * 60)
        raise
