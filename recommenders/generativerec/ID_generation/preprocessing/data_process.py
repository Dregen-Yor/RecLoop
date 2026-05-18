"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ast
import gzip
import html
import json
import logging
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import requests
import tqdm
try:
    import wget
except ImportError:
    wget = None



def setup_logging(dataset_name):
    """Configure logging system"""
    log_dir = "./ID_generation/preprocessing/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"{dataset_name}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
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


def parse_jsonl(path):
    """Parse uncompressed JSONL file"""
    logger = logging.getLogger(__name__)
    logger.info(f"Start parsing file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except Exception as e:
        logger.error(f"Failed to parse file: {path}, error: {e}")
        raise


def parse(path):  # for Amazon
    g = gzip.open(path, "r")
    for l in g:
        l = l.replace(b"true", b"True").replace(b"false", b"False")
        yield eval(l)


def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"Failed to download {os.path.basename(path)}")


class Amazon:
    def __init__(self, root, dataset_name, rating_score, use_local_json=False):
        self.root = os.path.abspath(root)
        self.dataset_name = dataset_name
        self.rating_score = rating_score
        self.use_local_json = use_local_json
        
        
        if not use_local_json:
            self.download()
        self.datas = self.process()

    def download(self):
        path = self.root
        os.makedirs(path, exist_ok=True)

        url_dict = {
            "Beauty": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
            ],
            "Toys_and_Games": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
            ],
            "Sports_and_Outdoors": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
            ],
            "Office_Products": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz",
            ],
        }
        # we save the raw data into `directory`
        for url in url_dict[self.dataset_name]:
            # Extract the filename from the URL
            filename = url.split("/")[-1]
            filepath = os.path.join(path, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found, downloading...")
                download_file(url, filepath)

    def process(self):
        """
        Process review data
        
        Local JSON format fields (2023 version):
        - user_id: User ID
        - parent_asin: Item ID
        - timestamp: Timestamp (ms)
        - rating: Rating
        
        Original format fields (old version):
        - reviewerID: User ID
        - asin: Item ID
        - unixReviewTime: Timestamp (sec)
        - overall: Rating
        """
        logger = logging.getLogger(__name__)
        datas = []
        
        if self.use_local_json:

            data_file = f"../../../data/{self.dataset_name}/{self.dataset_name}.json"
            logger.info(f"Reading local JSON file: {data_file}")
            
            if not os.path.exists(data_file):
                logger.error(f"File does not exist: {data_file}")
                return datas
            
            line_count = 0
            filtered_by_rating = 0
            filtered_by_user = 0
            
            for inter in parse_jsonl(data_file):
                line_count += 1
                
                user = inter.get("user_id")
                item = inter.get("parent_asin") or inter.get("asin")
                rating = inter.get("rating", inter.get("overall", 0))
                timestamp = inter.get("timestamp", inter.get("unixReviewTime", 0))
                
                
                if float(rating) <= self.rating_score:
                    filtered_by_rating += 1
                    continue
                
                if timestamp > 10000000000:
                    timestamp = timestamp // 1000
                
                datas.append((user, item, int(timestamp)))
                
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count} reviews, kept {len(datas)}")
            
            logger.info(f"Review processing completed: total={line_count}, kept={len(datas)}, filtered_by_rating={filtered_by_rating}, filtered_by_user={filtered_by_user}")
        else:

            data_file = os.path.join(
                self.root, f"reviews_{self.dataset_name}_5.json.gz"
            )
            logger.info(f"Reading compressed file: {data_file}")
            
            for inter in parse(data_file):
                if float(inter["overall"]) <= self.rating_score:
                    continue
                user = inter["reviewerID"]
                item = inter["asin"]
                time = inter["unixReviewTime"]
                datas.append((user, item, int(time)))
        
        return datas

    def process_meta(self, data_maps):
        """
        Process item metadata
        
        Local JSON format fields (2023 version):
        - parent_asin: Item ID
        - title: Item title
        - description: Item description
        - price: Price
        - categories: Category list
        - details: Details
        
        Original format fields (old version):
        - asin: Item ID
        - title: Item title
        - description: Item description
        - price: Price
        - categories: Category list
        - brand: Brand
        """
        logger = logging.getLogger(__name__)
        datas = {}
        
        if self.use_local_json:

            meta_file = f"../../../data/{self.dataset_name}/{self.dataset_name}_metadata.json"
            logger.info(f"Reading local metadata file: {meta_file}")
            
            if not os.path.exists(meta_file):
                logger.error(f"Metadata file does not exist: {meta_file}")
                return datas
            
            item_asins = set(data_maps["item2id"].keys())
            line_count = 0
            matched_count = 0
            
            for info in parse_jsonl(meta_file):
                line_count += 1
                
                
                asin = info.get("parent_asin") or info.get("asin")
                
                if asin not in item_asins:
                    continue
                
                
                unified_info = {
                    "asin": asin,
                    "title": info.get("title", ""),
                    "description": info.get("description", []),
                    "price": info.get("price", info.get("price", "")),
                    "categories": info.get("categories", []),
                    "brand": info.get("brand", info.get("details", {}).get("Brand", "")),
                    "details": info.get("details", {}),
                }
                
                datas[asin] = unified_info
                matched_count += 1
                
                if line_count % 1000 == 0:
                    logger.info(f"Processed {line_count} metadata entries, matched {matched_count} items")
            
            logger.info(f"Metadata processing completed: total={line_count}, matched={matched_count}")
        else:

            meta_file = os.path.join(
                self.root, f"meta_{self.dataset_name}.json.gz"
            )
            logger.info(f"Reading compressed metadata file: {meta_file}")
            
            item_asins = set(data_maps["item2id"].keys())
            for info in parse(meta_file):
                if info["asin"] not in item_asins:
                    continue
                datas[info["asin"]] = info
        
        return datas


class Steam:
    def __init__(self, root, user_core) -> None:
        self.root = os.path.abspath(root)
        self.urls = {
            "reviews": "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
            "games": "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
        }
        self.download()
        self.process(user_core)

    def download(self):
        path = os.path.join(self.root, "steam")

        if os.path.exists(os.path.join(path, "steam_reviews.json")) and os.path.exists(
            os.path.join(path, "steam_games.json")
        ):
            print(f"{path} exists, download is not needed.")
            return

        os.makedirs(path, exist_ok=True)
        for d in ["games", "reviews"]:
            print(f"downloading steam from {self.urls[d]}")
            file_name = wget.download(self.urls[d], out=path)
            content = gzip.open(file_name, "rb")
            content = content.read().decode("utf-8").split("\n")
            content = [
                json.loads(json.dumps(ast.literal_eval(line)))
                for idx, line in enumerate(content)
                if line
            ]
            with open(file_name[:-3], "w") as f:
                json.dump(content, f)

    def process(self, user_core):
        path = os.path.join(self.root.replace("raw_data", "processed"))
        os.makedirs(path, exist_ok=True)

        print(f"preprocessing steam ...")
        review_file = glob(f"{os.path.join(self.root, 'steam')}/steam_reviews.json")[0]

        with open(review_file, "r") as f:
            raw_data = json.load(f)

        user_counts = Counter([entry["username"] for entry in raw_data])
        raw_data = [
            entry for entry in raw_data if user_counts[entry["username"]] >= user_core
        ]

        user_id, item_id = 1, 1
        self.user2id, self.item2id, self.id2user, self.id2item = {}, {}, {}, {}
        self.item2id["<pad>"], self.id2item[0] = 0, "<pad>"
        self.item2review = {}
        for entry in tqdm.tqdm(raw_data, desc="Mapping unique users and items ..."):
            if entry["username"] not in self.user2id:
                self.user2id[entry["username"]] = user_id
                self.id2user[user_id] = entry["username"]
                user_id += 1
            if entry["product_id"] not in self.item2id:
                self.item2id[entry["product_id"]] = item_id
                self.id2item[item_id] = entry["product_id"]
                self.item2review[item_id] = entry["text"]
                item_id += 1

        self.sequence_raw = []
        for entry in tqdm.tqdm(raw_data, desc="Constructing sequence and graph ..."):
            self.sequence_raw.append(
                (
                    self.user2id[entry["username"]],
                    self.item2id[entry["product_id"]],
                    int(datetime.fromisoformat(entry["date"]).timestamp()),
                )
            )
        # the item in sequence_raw is in item_id domain

    def process_meta_infos(self):
        meta_file = glob(f"{os.path.join(self.root, 'steam')}/steam_games.json")[0]
        with open(meta_file, "r") as f:
            raw_data = json.load(f)

        items = {}
        for entry in tqdm.tqdm(raw_data, desc="Creating item content features...."):
            if "id" in entry and entry["id"] in self.item2id:
                meta_dict = {}
                meta_dict["title"] = f"{entry['title']}"
                meta_dict["genre"] = (
                    f"{' '.join(entry['genres']) if 'genres' in entry else 'Unknown'}"
                )
                meta_dict["tags"] = (
                    f"{' '.join(entry['tags']) if 'tags' in entry else 'Unknown'}"
                )
                meta_dict["specs"] = (
                    f"{' '.join(entry['specs']) if 'specs' in entry else 'Unknown'}"
                )
                meta_dict["price"] = f"{entry.get('price', 0)}"
                meta_dict["publisher"] = f"{entry.get('publisher', 'Unknown')}"
                meta_dict["sentiment"] = f"{entry.get('sentiment', 'Unknown')}"
                items[self.item2id[entry["id"]]] = (
                    meta_dict  # id is product_id, item2id maps product_id to item_id that starts from 1
                )

        return items


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(user_items, user_core, item_core):
    """Iteratively filter user-item interactions until core conditions are met (improved version)"""
    logger = logging.getLogger(__name__)
    logger.info(f"Start filtering: user interaction threshold={user_core}, item occurrence threshold={item_core}")
    start_time = datetime.now()

    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    iteration = 0

    while not isKcore:
        iteration += 1
        logger.info(f"Round {iteration} filtering...")
        
        
        prev_user_count = len(user_items)
        prev_total_interactions = sum(len(items) for items in user_items.values())
        
        
        to_delete_users = [user for user in user_items if user_count[user] < user_core]
        logger.info(f"Number of users to remove: {len(to_delete_users)}")
        
        
        for user in to_delete_users:
            user_items.pop(user)
        
        
        for user in list(user_items.keys()):
            filtered_items = [item for item in user_items[user] if item_count[item] >= item_core]
            if filtered_items:
                user_items[user] = filtered_items
            else:
                user_items.pop(user)
        
        
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
        
        
        current_user_count = len(user_items)
        current_total_interactions = sum(len(items) for items in user_items.values())

        logger.info(f"  Removed users: {prev_user_count - current_user_count}")
        logger.info(f"  Current users: {current_user_count}, items: {len(item_count)}")
        logger.info(f"  Current total interactions: {current_total_interactions}")
        
        
        if iteration > 50:
            logger.warning("Too many filtering iterations, possible issue")
            break

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Filtering completed! final users: {len(user_items)}, items: {len(set(item for items in user_items.values() for item in items))}")
    logger.info(f"Filtering time: {processing_time:.2f} s")
    return user_items


def remove_consecutive_duplicates(user_items):
    """Remove consecutive duplicate items in each user sequence, keep only one"""
    logger = logging.getLogger(__name__)
    logger.info("Start removing consecutive duplicate items...")
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
    
    return user_items


def get_attribute_Amazon(meta_infos, datamaps, attribute_core):
    category_key = "categories"

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info[category_key]:
            for cate in cates[1:]:
                attributes[cate] += 1
        try:
            attributes[info["brand"]] += 1
        except:
            pass

    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info["brand"]] >= attribute_core:
                new_meta[iid].append(info["brand"])
        except:
            pass
        for cates in info[category_key]:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, item_attributes in new_meta.items():
        item_id = datamaps["item2id"][iid]
        items2attributes[item_id] = []
        for attribute in item_attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )
    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    datamaps["attributeid2num"] = attributeid2num
    return (
        len(attribute2id),
        np.mean(attribute_lens),
        datamaps,
        items2attributes,
        attributes,
    )


def get_attribute_steam(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            attributes[info["genre"]] += 1
        except:
            pass
    print(f"before delete, attribute num:{len(attributes)}")
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info["genre"]] >= attribute_core:
                new_meta[iid].append(info["genre"])
        except:
            pass

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        try:
            item_id = datamaps["item2id"][iid]
        except:
            continue

        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))

    print(f"after delete, attribute num:{len(attribute2id)}")
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )
    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def clean_text2(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = " ".join(raw_text[0])
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


def meta_map(
    meta_infos,
    item2id,
    attributes,
    features_needed=["title", "price", "brand", "categories", "description"],
    attribute_core=0,
    prompt_format="v1",
):
    id2meta = {}
    item2meta = {}

    for item, meta in tqdm.tqdm(meta_infos.items()):
        meta_text = ""
        keys = set(meta.keys()).intersection(features_needed)

        if prompt_format == "amazon":
            if "title" in keys and meta["title"] != "":
                meta_text += f"This item is called '{feature_process(meta['title'])}'. "
            if "price" in keys and meta["price"] is not None:
                meta_text += f"It is priced at {feature_process(meta['price'])} "
            if (
                "brand" in keys
                and meta["brand"] != ""
                and attributes[meta["brand"]] >= attribute_core
            ):
                meta_text += (
                    f"It is manufactured by '{feature_process(meta['brand'])}'. "
                )
            if "categories" in keys:
                meta_text += f"It belongs to the categories of {feature_process(meta['categories'])}. "
            if "description" in keys:
                meta_text += f"The description of this item is: {feature_process(meta['description'])}. "
        elif prompt_format == "unisrec":
            for meta_key in features_needed:
                if meta_key in meta:
                    meta_value = clean_text2(meta[meta_key])
                    meta_text += meta_value + " "

        elif prompt_format == "steam":
            # Title
            if "title" in keys and meta["title"] != "":
                meta_text += f"This game is called '{meta['title']}'. "
            if "publisher" in keys and meta["publisher"] != "":
                meta_text += f"Developed by {meta['publisher']}, "
            if "price" in keys and meta["price"] is not None:
                meta_text += f"this game is priced at {meta['price']}. "
            if "specs" in keys:
                specs_list = meta["specs"].split(" ")
                formatted_specs = ", ".join(specs_list)
                meta_text += f"This game features a variety of gameplay modes including {formatted_specs}. "
            if "sentiment" in keys:
                meta_text += (
                    f"The game holds a {meta['sentiment']} sentiment among players. "
                )
            if "tags" in keys:
                meta_text += f"Tags of the game include: {meta['tags']}. "

        if (
            prompt_format == "steam"
        ):  # for steam, the key of the meta_infos are already passed in to item2id, so it is essentially id
            try:
                id = item2id[item]
            except:
                continue
            id2meta[id] = meta_text
        else:
            item2meta[item] = meta_text
            id = item2id[item]
            id2meta[id] = meta_text
    return id2meta


def get_item_review_map(review_mapping, data_maps, meta_infos):
    id2review = defaultdict(dict)
    for reviewer_id, items in review_mapping.items():
        try:
            user_id = data_maps["user2id"][reviewer_id]
            for item, review in items.items():
                id = data_maps["item2id"][item]
                title = (
                    "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
                )
                categories = (
                    meta_infos[item]["categories"]
                    if "categories" in meta_infos[item]
                    else ""
                )
                id2review[user_id][id] = (
                    title,
                    categories,
                ) + review
        except:
            pass

    return id2review


def get_steam_reviews(user_seq, review_mapping, datamaps, meta_infos):
    id2review = defaultdict(list)
    for reviewer_id in user_seq.keys():
        user_id = datamaps["user2id"][reviewer_id]
        for item in user_seq[reviewer_id]:
            id = datamaps["item2id"][item]
            title = "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
            genre = meta_infos[item]["genre"] if "genre" in meta_infos[item] else ""
            tags = meta_infos[item]["tags"] if "tags" in meta_infos[item] else ""
            genre = meta_infos[item]["specs"] if "specs" in meta_infos[item] else ""
            price = meta_infos[item]["price"] if "price" in meta_infos[item] else ""
            publisher = (
                meta_infos[item]["publisher"] if "publisher" in meta_infos[item] else ""
            )
            sentiment = (
                meta_infos[item]["sentiment"] if "sentiment" in meta_infos[item] else ""
            )
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "title": title,
                    "tags": tags,
                    "genre": genre,
                    "price": price,
                    "publisher": publisher,
                    "sentiment": sentiment,
                    "review": review_mapping[item],
                }
            )
    return id2review


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ""
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ","
    return res_num[:-1]


def id_map(user_items):  # user_items dict
    """
    Create user and item ID mapping (improved version)
    
    Key improvements:
    1. Use sorted() to ensure mapping stability and reproducibility
    2. Map IDs to integer type (starting from 1)
    3. Also maintain string format mapping (for compatibility with existing code)
    
    This ensures complete consistency with process2023_nosplit.py ID mapping
    """
    logger = logging.getLogger(__name__)
    logger.info("Start creating user and item mappings...")
    
    
    valid_items = set()
    for items in user_items.values():
        valid_items.update(items)
    
    logger.info(f"Valid users: {len(user_items)}")
    logger.info(f"Valid items: {len(valid_items)}")
    
    
    sorted_users = sorted(user_items.keys())
    sorted_items = sorted(valid_items)
    
    
    user_mapping_int = {user_id: idx for idx, user_id in enumerate(sorted_users, start=1)}
    item_mapping_int = {item_id: idx for idx, item_id in enumerate(sorted_items, start=1)}
    
    logger.info(f"User ID range: 1 ~ {len(user_mapping_int)}")
    logger.info(f"Item ID range: 1 ~ {len(item_mapping_int)}")
    logger.info(f"First 5 user mappings: {list(user_mapping_int.items())[:5]}")
    logger.info(f"First 5 item mappings: {list(item_mapping_int.items())[:5]}")
    
    
    user2id = {user: str(idx) for user, idx in user_mapping_int.items()}
    item2id = {item: str(idx) for item, idx in item_mapping_int.items()}
    id2user = {str(idx): user for user, idx in user_mapping_int.items()}
    id2item = {str(idx): item for item, idx in item_mapping_int.items()}
    
    
    final_data = {}
    for user, items in user_items.items():
        uid = user2id[user]
        iids = [item2id[item] for item in items]
        final_data[uid] = iids
    
    data_maps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        
        "user_mapping_int": user_mapping_int,
        "item_mapping_int": item_mapping_int,
    }
    
    return final_data, len(user_mapping_int), len(item_mapping_int), data_maps


def get_interaction(datas, meta_data_set=None):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if meta_data_set is not None and item not in meta_data_set:
            continue
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(", ".join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text


def feature_process(feature):
    sentence = ""
    if feature is None:
        return ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += "."
    elif isinstance(feature, (list, tuple)) and len(feature) > 0 and isinstance(feature[0], list):
        # this should be the categories
        for v1 in feature:
            for v in v1[1:]:
                sentence += clean_text(v)
                sentence += ", "
        sentence = sentence[:-2]
        # sentence += "."
    elif isinstance(feature, list):
        for v1 in feature:
            sentence += clean_text(v1)
    else:
        sentence = clean_text(feature)
    return sentence


def preprocessing(config):
    dataset_name = config["name"]
    data_file, id2meta_file, item2attribute_file = preprocessing_each_dataset(
        config, dataset_name
    )
    return data_file, id2meta_file, item2attribute_file


def preprocessing_each_dataset(config, dataset_name):

    logger = setup_logging(dataset_name)
    logger.info("=" * 60)
    logger.info(f"Start processing dataset: {dataset_name}")
    logger.info("=" * 60)
    
    data_type, features_needed = config["type"], config["features_needed"]
    prompt_format = config["prompt_format"]
    features_used = "_".join(features_needed)
    raw_data_path = config["raw_data_path"]
    # by default this is "./ID_generation/preprocessing/raw_data/"
    processed_data_path = config["processed_data_path"]
    # by default this is "./ID_generation/preprocessing/processed/"
    
    
    use_local_json = config.get("use_local_json", False)

    # a list of file paths we will use
    data_file = os.path.join(processed_data_path, f"{dataset_name}.txt")
    id2meta_file = os.path.join(
        processed_data_path,
        f"{dataset_name}_{features_used}_{prompt_format}_id2meta.json",
    )
    item2attributes_file = os.path.join(
        processed_data_path, f"{dataset_name}_item2attributes.json"
    )

    user_mapping_file = os.path.join(processed_data_path, f"{dataset_name}_user_mapping.npy")
    item_mapping_file = os.path.join(processed_data_path, f"{dataset_name}_item_mapping.npy")

    logger.info(f"data_name: {dataset_name}, data_type: {data_type}")
    logger.info(f"use_local_json: {use_local_json}")

    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0
    min_interactions = 3
    logger.info(f"Use user core: {user_core}, item core: {item_core}, min interactions: {min_interactions}")

    if data_type == "Amazon":
        dataset = Amazon(
            raw_data_path, 
            dataset_name, 
            rating_score,
            use_local_json=use_local_json
        )
        datas = dataset.datas
    elif data_type == "steam":
        dataset = Steam(raw_data_path, user_core)
        datas = dataset.sequence_raw
    else:
        raise NotImplementedError

    meta_data_set = None
    user_items = get_interaction(datas, meta_data_set)
    # raw_id user: [item1, item2, item3...]
    
    logger.info(f"Initial users: {len(user_items)}")
    logger.info(f"Initial total interactions: {sum(len(items) for items in user_items.values())}")

    # filter K-core
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    logger.info(f"User {user_core}-core complete! Item {item_core}-core complete!")
    
    
    user_items = remove_consecutive_duplicates(user_items)
    
    
    logger.info(f"Start filtering by minimum interactions (>= {min_interactions})...")
    before_count = len(user_items)
    user_items = {user: items for user, items in user_items.items() if len(items) >= min_interactions}
    after_count = len(user_items)
    logger.info(f"Minimum interaction filtering completed: before={before_count}, after={after_count}, removed={before_count - after_count}")
    
    
    user_items_id, user_num, item_num, data_maps = id_map(user_items)
    user_count, item_count, _ = check_Kcore(
        user_items_id, user_core=user_core, item_core=item_core
    )

    # calculate sparsity
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = (
        np.mean(user_count_list),
        np.min(user_count_list),
        np.max(user_count_list),
    )
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = (
        np.mean(item_count_list),
        np.min(item_count_list),
        np.max(item_count_list),
    )
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    seqs_length = [len(user_items_id[i]) for i in user_items_id.keys()]
    show_info = (
        f"Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n"
        + f"Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n"
        + f"Interaction Num: {interact_num}, Sparsity: {sparsity:.2f}%\n"
        + f"Sequence Length Mean: {(sum(seqs_length) / len(seqs_length)):.2f}, Median: {statistics.median(seqs_length)}"
    )
    print(show_info)

    print("Begin extracting meta infos...")

    if data_type == "Amazon":
        meta_infos = dataset.process_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes, attributes = (
            get_attribute_Amazon(meta_infos, data_maps, attribute_core)
        )
        id2meta = meta_map(
            meta_infos,
            data_maps["item2id"],
            attributes,
            features_needed,
            attribute_core,
            prompt_format,
        )
        # item2review = get_item_review_map(review_mapping, data_maps, meta_infos)
    elif data_type == "steam":
        meta_infos = dataset.process_meta_infos()  # key is in item_id
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_steam(
            meta_infos, data_maps, attribute_core
        )
        id2meta = meta_map(
            meta_infos,
            data_maps["item2id"],
            None,
            features_needed,
            attribute_core,
            prompt_format,
        )

    logger.info(
        f"{dataset_name} & {add_comma(user_num)} & {add_comma(item_num)} & {user_avg:.1f}"
        f" & {item_avg:.1f} & {add_comma(interact_num)} & {sparsity:.2f}% & {add_comma(attribute_num)} &"
        f" {avg_attribute:.1f} \\\\"
    )

    # -------------- Save Data ---------------
    logger.info("Start saving data files...")
    
    
    logger.info(f"Saving interaction sequence: {data_file}")
    user_mapping_int = data_maps["user_mapping_int"]
    item_mapping_int = data_maps["item_mapping_int"]
    
    
    
    users_written = 0
    total_interactions_written = 0
    with open(data_file, "w") as out:
        for original_user_id, items in user_items.items():
            new_user_id = user_mapping_int[original_user_id]
            new_items = [item_mapping_int[item] for item in items]
            out.write(f"{new_user_id} {' '.join(str(item) for item in new_items)}\n")
            users_written += 1
            total_interactions_written += len(new_items)
    
    logger.info(f"Interaction sequence saved: users={users_written}, total interactions={total_interactions_written}")


    logger.info(f"Saving metadata: {id2meta_file}")
    json_str = json.dumps(id2meta)
    with open(id2meta_file, "w") as out:
        out.write(json_str)


    logger.info(f"Saving item attributes: {item2attributes_file}")
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, "w") as out:
        out.write(json_str)
    
    
    logger.info(f"Saving user mapping: {user_mapping_file}")
    np.save(user_mapping_file, user_mapping_int)
    logger.info(f"Saving item mapping: {item_mapping_file}")
    np.save(item_mapping_file, item_mapping_int)
    
    logger.info("=" * 60)
    logger.info(f"Data processing completed!")
    logger.info(f"Output files:")
    logger.info(f"  - Interaction sequence: {data_file}")
    logger.info(f"  - Metadata: {id2meta_file}")
    logger.info(f"  - Item attributes: {item2attributes_file}")
    logger.info(f"  - User mapping: {user_mapping_file}")
    logger.info(f"  - Item mapping: {item_mapping_file}")
    logger.info("=" * 60)

    return data_file, id2meta_file, item2attributes_file
