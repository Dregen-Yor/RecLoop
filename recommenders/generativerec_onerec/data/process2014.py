import argparse
import json
import gzip
import os
import re
import html
import ast
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


def setup_logging(args):
    """Configure logging system"""
    if os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}", exist_ok=True)
    else:
        os.makedirs(f"./{args.dataset}", exist_ok=True)

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


def clean_text(text):
    """Clean text: remove HTML tags and extra whitespace"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_path(path):
    """Create directory"""
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """Write JSON file"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """Write mapping file (tab-separated)"""
    with open(file_path, "w") as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


def parse_jsonl_gz(path):
    """Parse compressed JSONL file, supporting standard JSON and Python dict format (single quotes)"""
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
                        logger.warning(f"Line {line_num} parsing failed: {e}")
                        continue
    except Exception as e:
        logger.error(f"Failed to parse compressed file: {path}, error: {e}")
        raise


def load_sampled_user_ids(path):
    """Load sampled user file, return user ID set. Return None if file doesn't exist."""
    logger.info(f"Start loading sampled user file: {path}")
    
    if not path or (not os.path.exists(path)):
        logger.warning(f"Sampled user file not found: {path}, processing all users")
        return None
    
    users = set()
    count_lines = 0
    
    try:
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                count_lines += 1
                try:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        obj = json.loads(line_stripped)
                    except json.JSONDecodeError:
                        obj = ast.literal_eval(line_stripped)
                    uid = obj.get('user_id')
                    if uid:
                        users.add(uid)
                except Exception as e:
                    logger.warning(f"Failed parsing line {count_lines}: {e}")
                    continue
        
                    logger.info(f"Successfully loaded sampled users: {len(users)} (from {count_lines} lines)")
        return users
        
    except Exception as e:
        logger.error(f"Failed to load sampled user file: {e}")
        return None


def convert_ms_to_sec(ts_ms):
    """Amazon14 sort_timestamp is in milliseconds, convert to seconds"""
    try:
        return int(ts_ms // 1000)
    except:
        return 0


def load_reviews_amazon14(reviews_file):
    """Load Amazon14 review data"""
    if not os.path.exists(reviews_file):
        logger.error(f"Review file does not exist: {reviews_file}")
        return []

    reviews = []
    logger.info(f"Start loading Amazon14 review file: {reviews_file}")
    
    if reviews_file.endswith('.gz'):
        data_gen = parse_jsonl_gz(reviews_file)
    else:
        logger.error("A .gz review file is required")
        return []

    for r in tqdm(data_gen, desc="Reading reviews"):
        try:
            item_id = r.get("asin")
            if item_id is None:
                continue
            ts = convert_ms_to_sec(r.get("unixReviewTime", 0))
            review_obj = {
                "user_id": r["reviewerID"],
                "asin": item_id,
                "rating": float(r.get("overall", 0)),
                "review_title": clean_text(r.get("summary", "")),
                "review_text": clean_text(r.get("reviewText", "")),
                "timestamp": ts
            }
            reviews.append(review_obj)
        except Exception:
            continue

    logger.info(f"Loading completed: {len(reviews)} reviews")
    return reviews

def load_metadata_amazon14(metadata_file):
    """Load Amazon14 metadata"""
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file does not exist: {metadata_file}")
        return {}, {}

    logger.info(f"Start loading Amazon14 metadata: {metadata_file}")

    asin2meta = {}
    asin2title = {}

    if metadata_file.endswith('.gz'):
        data_gen = parse_jsonl_gz(metadata_file)
    else:
        logger.error("A .gz metadata file is required")
        return {}, {}

    for m in tqdm(data_gen, desc="Reading metadata"):
        try:
            asin = m.get("asin", None)
            if asin is None:
                continue
            title = clean_text(m.get("title", ""))
            asin2meta[asin] = {
                "title": title,
                "price": m.get("price", None),
                "categories": m.get("categories", []),
                "description": m.get("description", []),
                "brand": m.get("brand", "")}
            if len(title) > 0:
                asin2title[asin] = title
        except Exception:
            continue

    logger.info(f"Loading completed: {len(asin2meta)} metadata entries, {len(asin2title)} titles")
    return asin2meta, asin2title


def k_core_filter_amazon14(reviews, K=5):
    """K-core filtering (without time filtering and title filtering)"""
    logger.info(f"Start K-core filtering, K={K}")

    remove_users = set()
    remove_items = set()

    while True:
        user_counts = {}
        item_counts = {}
        new_reviews = []
        changed = False
        kept_total = 0

        for r in tqdm(reviews, desc="K-core iteration"):
            user = r["user_id"]
            item = r["asin"]

            if user in remove_users:
                continue
            if item in remove_items:
                continue

            user_counts[user] = user_counts.get(user, 0) + 1
            item_counts[item] = item_counts.get(item, 0) + 1

            new_reviews.append(r)
            kept_total += 1

        for u, c in user_counts.items():
            if c < K:
                remove_users.add(u)
                changed = True

        for i, c in item_counts.items():
            if c < K:
                remove_items.add(i)
                changed = True

        logger.info(
            f"[k-core] Users={len(user_counts)}, Items={len(item_counts)}, "
            f"Reviews={kept_total}, Density={kept_total / (max(1, len(user_counts) * len(item_counts))):.6f}"
        )

        if not changed:
            break

        reviews = new_reviews

    logger.info("K-core filtering completed")
    return new_reviews, user_counts, item_counts


def remove_consecutive_duplicates_reviews(reviews):
    """
    Remove consecutive duplicate items in each user's review sequence
    Performed before creating index mappings to reduce item count
    """
    logger.info("Starting to remove consecutive duplicate items...")
    
    user_reviews = defaultdict(list)
    for r in reviews:
        user_reviews[r["user_id"]].append(r)
    
    deduplicated_reviews = []
    total_before = len(reviews)
    
    for user_id, user_revs in tqdm(user_reviews.items(), desc="Removing consecutive duplicates"):
        user_revs.sort(key=lambda x: x["timestamp"])
        prev_item = None
        for r in user_revs:
            if r["asin"] != prev_item:
                deduplicated_reviews.append(r)
                prev_item = r["asin"]
    
    total_after = len(deduplicated_reviews)
    removed_count = total_before - total_after
    
    logger.info(f"[Dedup] Reviews before dedup: {total_before}")
    logger.info(f"[Dedup] Reviews after dedup: {total_after}")
    logger.info(f"[Dedup] Removed consecutive duplicates: {removed_count}")
    
    return deduplicated_reviews


def convert_interactions_amazon14(reviews):
    """
    Convert interaction data to amazon18 style:
    - user2index (starting from 1)
    - item2index (starting from 1)
    - user2items[user_index] = [item_index...]
    - interactions list
    """
    user_reviews = defaultdict(list)

    for r in reviews:
        user_reviews[r["user_id"]].append(r)

    for u in user_reviews:
        user_reviews[u].sort(key=lambda x: x["timestamp"])

    all_users = set(user_reviews.keys())
    all_items = set()
    for user_revs in user_reviews.values():
        for r in user_revs:
            all_items.add(r["asin"])
    
    sorted_users = sorted(all_users)
    sorted_items = sorted(all_items)
    
    user2index = {user_id: idx for idx, user_id in enumerate(sorted_users, start=1)}
    item2index = {item_id: idx for idx, item_id in enumerate(sorted_items, start=1)}
    
    logger.info(f"[Before mapping] users={len(sorted_users)}, items={len(sorted_items)}")

    user2items = defaultdict(list)
    interactions = []

    for user in user_reviews:
        uid = user2index[user]
        for r in user_reviews[user]:
            item = r["asin"]
            iid = item2index[item]
            user2items[uid].append(iid)
            interactions.append((user, item, r["rating"], r["timestamp"]))

    logger.info(
        f"[Mapping] Users={len(user2index)}, Items={len(item2index)}, "
        f"Interactions={len(interactions)}"
    )
    return user2items, user2index, item2index, interactions


# def build_interaction_list_amazon14(reviews, user2index, item2index, asin2title):
#     """


#         [user_id_original,
#          history_asins,
#          target_asin,
#          history_item_ids,
#          target_item_id,
#          history_titles,
#          target_title,
#          history_ratings,
#          target_rating,
#          history_timestamps,
#          target_timestamp]
#     """

#     interact = {}


#         u = r["user_id"]
#         i = r["asin"]

#         if u not in interact:
#             interact[u] = {
#                 "items": [],
#                 "ratings": [],
#                 "timestamps": [],
#                 "item_ids": [],
#                 "titles": [],
#             }

#         interact[u]["items"].append(i)
#         interact[u]["ratings"].append(r["rating"])
#         interact[u]["timestamps"].append(r["timestamp"])
#         interact[u]["item_ids"].append(item2index[i])
#         interact[u]["titles"].append(asin2title.get(i, ""))


#     interaction_list = []


#         items = interact[u]["items"]
#         ratings = interact[u]["ratings"]
#         timestamps = interact[u]["timestamps"]
#         item_ids = interact[u]["item_ids"]
#         titles = interact[u]["titles"]


#         all_data = list(zip(items, ratings, timestamps, item_ids, titles))
#         all_data.sort(key=lambda x: x[2])
#         items, ratings, timestamps, item_ids, titles = zip(*all_data)

#         items = list(items)
#         ratings = list(ratings)
#         timestamps = list(timestamps)
#         item_ids = list(item_ids)
#         titles = list(titles)

#         for i in range(1, len(items)):
#             st = max(i - 10, 0)

#             interaction_list.append([











#             ])


#     interaction_list.sort(key=lambda x: int(x[-1]))


#     return interaction_list


def build_item_features_amazon14(asin2meta, item2index):
    """
    Convert Amazon14 metadata to item feature dictionary:
        - title
        - description (list → str)
        - features
        - categories
        - brand/store
        - details
        - images
    """
    item2feature = {}

    for asin, idx in item2index.items():
        m = asin2meta.get(asin, {})

        title = clean_text(m.get("title", ""))

        desc = m.get("description", [])
        if isinstance(desc, list):
            desc = " ".join([clean_text(d) for d in desc])
        else:
            desc = clean_text(desc)

        # # features: list
        # feats = m.get("features", [])
        # feats = " ".join([clean_text(f) for f in feats]) if isinstance(feats, list) else clean_text(str(feats))

        # categories: list of strings → join
        cats = m.get("categories", [])
        if isinstance(cats, list):
            cats = ", ".join([clean_text(c) for c in cats])
        else:
            cats = clean_text(str(cats))

        # brand/store
        brand = clean_text(m.get("brand", ""))
        # details = m.get("details", {})

        # images = m.get("images", [])
        # image_urls = []
        # for img in images:
        #     if isinstance(img, dict):
        #         if "hi_res" in img:
        #             image_urls.append(img["hi_res"])
        #         elif "large" in img:
        #             image_urls.append(img["large"])
        #         elif "thumb" in img:
        #             image_urls.append(img["thumb"])

        item2feature[idx] = {
            "title": title,
            "description": desc,
            # "features": feats,
            "categories": cats,
            "brand": brand,
            # "details": details,
            # "images": image_urls,
        }

    logger.info(f"[Features] Built {len(item2feature)} item features")
    return item2feature


def build_review_data_amazon14(reviews, user2index, item2index):
    """
    Build review_data:
        key = (uid, iid, timestamp)
        value = {review, summary, helpful_votes, verified}
    """
    review_data = {}

    for r in tqdm(reviews, desc="Building review_data"):
        user = r["user_id"]
        asin = r["asin"]

        if user not in user2index or asin not in item2index:
            continue

        uid = user2index[user]
        iid = item2index[asin]
        ts = r["timestamp"]

        key = str((uid, iid, ts))

        review_data[key] = {
            "review": clean_text(r.get("review_text", "")),
            "summary": clean_text(r.get("summary", ""))}

    logger.info(f"[ReviewData] Stored {len(review_data)} reviews")
    return review_data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (folder name)")

    parser.add_argument("--user_k", type=int, default=5,
                        help="K-core threshold")

    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Metadata file path (meta_...jsonl.gz)")

    parser.add_argument("--reviews_file", type=str, default=None,
                        help="Review file path (...jsonl.gz)")

    parser.add_argument("--output_path", type=str, default=".",
                        help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()
    global logger
    logger = setup_logging(args)
    args.metadata_file = f"../../data/Amazon2014/meta_{args.dataset}.json.gz"
    args.reviews_file = f"../../data/Amazon2014/reviews_{args.dataset}_5.json.gz"
    logger.info("=" * 60)
    logger.info("Amazon 2014 data processing pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset         : {args.dataset}")
    logger.info(f"Metadata file    : {args.metadata_file}")
    logger.info(f"Review file      : {args.reviews_file}")
    logger.info(f"K-core threshold     : {args.user_k}")
    logger.info("=" * 60)

    total_start_time = datetime.now()

    try:
        asin2meta, asin2title = load_metadata_amazon14(args.metadata_file)
        reviews = load_reviews_amazon14(args.reviews_file)

        if len(reviews) == 0:
            logger.error("No review data loaded, aborting")
            return

        logger.info(f"Total reviews: {len(reviews)}")
        logger.info(f"Total metadata: {len(asin2title)}")

        unique_items = len(set(r["asin"] for r in reviews))
        logger.info(f"Unique items: {unique_items}")

        items_in_meta = sum(1 for r in reviews if r["asin"] in asin2title)
        logger.info(f"Reviews with metadata title: {items_in_meta}")

        logger.info("Step 3: K-core filtering")
        reviews_filtered, user_counts, item_counts = k_core_filter_amazon14(
            reviews, K=args.user_k
        )

        logger.info(f"[K-core result] Users={len(user_counts)}, Items={len(item_counts)}, Reviews={len(reviews_filtered)}")

        reviews_deduplicated = remove_consecutive_duplicates_reviews(reviews_filtered)

        user_interaction_counts = defaultdict(int)
        for r in reviews_deduplicated:
            user_interaction_counts[r["user_id"]] += 1
        
        reviews_final = [r for r in reviews_deduplicated if user_interaction_counts[r["user_id"]] >= 3]
        
        final_users = len(set(r["user_id"] for r in reviews_final))
        final_items = len(set(r["asin"] for r in reviews_final))
        logger.info(f"[After filtering] Users={final_users}, Items={final_items}, Reviews={len(reviews_final)}")

        user2items, user2index, item2index, interactions = convert_interactions_amazon14(
            reviews_final
        )

        item2feature = build_item_features_amazon14(asin2meta, item2index)

        review_data = build_review_data_amazon14(
            reviews_final, user2index, item2index
        )

        logger.info("Step 11: Save JSON and mapping files")
        out_dir = os.path.join(args.output_path, args.dataset)
        check_path(out_dir)

        logger.info(f"[Save] User interaction data: {len(user2items)} users")
        write_json_file(user2items, os.path.join(out_dir, f"{args.dataset}.inter.json"))
        
        logger.info(f"[Save] Item feature data: {len(item2feature)} items")
        write_json_file(item2feature, os.path.join(out_dir, f"{args.dataset}.item.json"))
        
        logger.info(f"[Save] Review data: {len(review_data)} reviews")
        write_json_file(review_data, os.path.join(out_dir, f"{args.dataset}.review.json"))

        logger.info(f"[Save] User mapping: {len(user2index)} users")
        write_remap_index(user2index, os.path.join(out_dir, f"{args.dataset}.user2id"))
        
        logger.info(f"[Save] Item mapping: {len(item2index)} items")
        write_remap_index(item2index, os.path.join(out_dir, f"{args.dataset}.item2id"))

        logger.info("Step 12: Save user sequence txt file")
        txt_file_path = os.path.join(out_dir, f"{args.dataset}.txt")
        users_written = 0
        total_interactions_written = 0
        
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            for user_idx, item_indices in user2items.items():
                f.write(f'{user_idx} {" ".join(str(item) for item in item_indices)}\n')
                users_written += 1
                total_interactions_written += len(item_indices)

        logger.info(f"[Save] User sequence file: {txt_file_path}")
        logger.info(f"[Save] Written users: {users_written}")
        logger.info(f"[Save] Written total interactions: {total_interactions_written}")
        total_end_time = datetime.now()
        total_processing_time = (total_end_time - total_start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("Data processing pipeline completed")
        logger.info(f"Total processing time: {total_processing_time:.2f} s ({total_processing_time/60:.2f} min)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
