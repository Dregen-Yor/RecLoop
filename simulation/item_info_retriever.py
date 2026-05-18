
import os
import json
import gzip
import logging
import pickle
import ast
import numpy as np
from typing import Dict, List, Optional, Set, Union
from collections import defaultdict

class ItemInfoRetriever:
    def __init__(self, dataset_path: str, dataset_name: str = "Beauty"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.item_info = {}
        self.categories = []
        self.item_mapping = {}
        self.reverse_mapping = {}
        self.cache_file = os.path.join(dataset_path, f'{dataset_name}_item_info_cache.pkl')
        self.mapping_file = None
        self._setup_logging()
        self._load_mappings()
        self._load_item_info()

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def parse_jsonl_gz(self, path: str):
        """Parse compressed JSONL file"""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.strip())

    def parse_jsonl(self, path: str):
        """Parse uncompressed JSONL file"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield self._parse_line(line.strip())

    def _parse_line(self, line: str):
        """Parse single line data, supporting multiple formats"""
        if not line.strip():
            return None

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            pass

        try:
            json_str = line.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

            self.logger.warning(f"Failed to parse line: {line[:100]}...")
        return None

    def _load_mappings(self):
        """Load item mapping files"""
        self.logger.info("Starting to load item mapping files")

        possible_mapping_files = [
            os.path.join(self.dataset_path, self.dataset_name, 'item_mapping.npy'),
            os.path.join(self.dataset_path, 'item_mapping.npy'),
            os.path.join(self.dataset_path, f'{self.dataset_name}_item_mapping.npy')
        ]

        for mapping_file in possible_mapping_files:
            if os.path.exists(mapping_file):
                try:
                    self.item_mapping = np.load(mapping_file, allow_pickle=True).item()
                    self.reverse_mapping = {v: k for k, v in self.item_mapping.items()}
                    self.mapping_file = mapping_file
                    self.logger.info(f"Loaded mapping file: {mapping_file}")
                    self.logger.info(f"Mapping count: {len(self.item_mapping)}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load mapping file {mapping_file}: {e}")
                    continue

                    self.logger.warning("No valid mapping file found, using original IDs")

    def _load_item_info(self):
        """Load item information from dataset"""
        self.logger.info("=" * 50)
        self.logger.info("Starting to load item data")
        self.logger.info("=" * 50)


        if self._is_cache_valid():
            self.logger.info("Valid cache file found, loading from cache...")
            self._load_from_cache()
            return


        metadata_file = self._find_metadata_file()
        if not metadata_file:
            self.logger.warning("Metadata file not found, ")
            self.item_info = {}
            self.categories = []
            return

            self.logger.info(f"founddatafile: {metadata_file}")


        try:
            if metadata_file.endswith('.gz'):
                metadata_generator = self.parse_jsonl_gz(metadata_file)
            else:
                metadata_generator = self.parse_jsonl(metadata_file)

            self._process_metadata(metadata_generator)


            self._save_to_cache()

            self.logger.info("dataLoading completed")
            self.logger.info(f"loaditems: {len(self.item_info)}")
            self.logger.info(f"category: {len(set(self.categories))}")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"loaddataerror: {e}")
            self.item_info = {}
            self.categories = []

    def _find_metadata_file(self) -> Optional[str]:
        """Find metadata file, prioritizing filtered files"""
        possible_files = [
            os.path.join(self.dataset_path,self.dataset_name, f'{self.dataset_name}_metadata.json'),

            # os.path.join(self.dataset_path, 'metadata.json'),
            # os.path.join(self.dataset_path, 'metadata.json.gz'),
        ]

        for file_path in possible_files:
            self.logger.info(f"Checking data file: {file_path}")
            if os.path.exists(file_path):
                self.logger.info(f"Found data file: {file_path}")
                return file_path

        return None

    def _process_metadata(self, metadata_generator):
        """Process metadata generator"""
        self.item_info = {}
        category_set = set()

        for metadata in metadata_generator:
            if metadata is None:
                continue

            # item_id = metadata.get('parent_asin') or metadata.get('asin')
            item_id = metadata.get('asin')
            if not item_id:
                continue

            item_data = metadata

            self.item_info[item_id] = item_data

            categories = item_data.get('categories', [])
            if categories:
                if isinstance(categories, list):
                    for category in categories:
                        if isinstance(category, str):
                            category_set.add(category)
                        elif isinstance(category, list):
                            category_set.update(category)

        self.categories = list(category_set)

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid"""
        if not os.path.exists(self.cache_file):
            return False

        metadata_file = self._find_metadata_file()
        if not metadata_file:
            return False

        cache_mtime = os.path.getmtime(self.cache_file)
        metadata_mtime = os.path.getmtime(metadata_file)

        return metadata_mtime <= cache_mtime

    def _load_from_cache(self):
        """Load data from cache"""

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.item_info = cache_data.get('item_info', {})
                self.categories = cache_data.get('categories', [])
                self.logger.info("Loaded from cache")
        except Exception as e:
            self.logger.warning(f"Cache loading failed: {e}")
            self.item_info = {}
            self.categories = []

    def _save_to_cache(self):
        """Save data to cache"""
        try:
            cache_data = {
                'item_info': self.item_info,
                'categories': self.categories
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                self.logger.info(f"Saved to cache: {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Cache saving failed: {e}")

    def get_item_info(self, item_id: Union[str, int], item_mapping: Optional[Dict[str, int]] = None) -> Optional[Dict]:
        """Get item information

        Args:
            item_id: Item ID, can be:
                     - Original ASIN string
                     - Integer ID (will be automatically converted to ASIN)
                     - String format of integer ID
            item_mapping: Item ID mapping table, external mapping is prioritized if provided

        Returns:
            Item information dictionary or None
        """
        original_id = self._resolve_item_id(item_id, item_mapping)
        if original_id:
            return self.item_info.get(original_id)
        return None

    def _resolve_item_id(self, item_id: Union[str, int], external_mapping: Optional[Dict[str, int]] = None) -> Optional[str]:
        """Resolve item ID, return original ASIN

        Args:
            item_id: Input item ID
            external_mapping: Externally provided mapping table

        Returns:
            Original ASIN or None
        """

        if isinstance(item_id, str):

            if item_id in self.item_info:
                return item_id


            try:
                item_id = int(item_id)
            except ValueError:

                return None


        if isinstance(item_id, int):

            if external_mapping:
                reverse_mapping = {v: k for k, v in external_mapping.items()}
                original_id = reverse_mapping.get(item_id)
                if original_id:
                    return original_id


            if self.reverse_mapping:
                original_id = self.reverse_mapping.get(item_id)
                if original_id:
                    return original_id



            item_ids = list(self.item_info.keys())
            if 1 <= item_id <= len(item_ids):
                return item_ids[item_id - 1]  # 1-based indexing

        return None

    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return self.categories.copy()

    def get_item_title(self, item_id: Union[str, int], item_mapping: Optional[Dict[str, int]] = None) -> str:
        """Get item title"""
        info = self.get_item_info(item_id, item_mapping)
        return info.get('title', '') if info else ''

    def get_item_categories(self, item_id: Union[str, int], item_mapping: Optional[Dict[str, int]] = None) -> List[str]:
        """Get item categories"""
        info = self.get_item_info(item_id, item_mapping)
        return info.get('categories', []) if info else []

    def get_item_asin(self, item_id: Union[str, int], item_mapping: Optional[Dict[str, int]] = None) -> str:
        """Get item ASIN"""
        info = self.get_item_info(item_id, item_mapping)
        if info:

            # return info.get('parent_asin') or info.get('asin', '')
            return info.get('asin', '')
        return ''

    def has_item_info(self, item_id: Union[str, int], item_mapping: Optional[Dict[str, int]] = None) -> bool:
        """Check if item info exists"""
        return self.get_item_info(item_id, item_mapping) is not None

    def get_item_id_from_asin(self, asin: str) -> Optional[int]:
        """Get integer ID from ASIN"""
        return self.item_mapping.get(asin)

    def get_asin_from_item_id(self, item_id: int) -> Optional[str]:
        """Get ASIN from integer ID"""
        return self.reverse_mapping.get(item_id)

    def get_batch_item_info(self, item_ids: List[Union[str, int]], item_mapping: Optional[Dict[str, int]] = None) -> List[Optional[Dict]]:
        """Get batch item information

        Args:
            item_ids: List of item IDs
            item_mapping: Item ID mapping table, use mapping if provided

        Returns:
            List of item information dictionaries, in the same order as input
        """
        return [self.get_item_info(item_id, item_mapping) for item_id in item_ids]

    def get_items_by_category(self, category: str, item_mapping: Optional[Dict[str, int]] = None, return_ids: bool = False) -> Union[List[Dict], List[Union[str, int]]]:
        """Get items by category

        Args:
            category: Category name
            item_mapping: Item ID mapping table
            return_ids: If True, return item ID list; otherwise return detailed info list

        Returns:
            List of items in this category (IDs or detailed info)
        """
        items = []
        for item_id, info in self.item_info.items():
            if info:
                categories = info.get('categories', [])

                category_found = False
                if isinstance(categories, list):
                    for cat in categories:
                        if isinstance(cat, str) and cat == category:
                            category_found = True
                            break
                        elif isinstance(cat, list):

                            if category in cat:
                                category_found = True
                                break
                elif isinstance(categories, str) and categories == category:
                    category_found = True

                if category_found:
                    if return_ids:

                        if item_mapping and item_id in item_mapping:
                            items.append(item_mapping[item_id])
                        else:

                            mapped_id = self.item_mapping.get(item_id)
                            items.append(mapped_id if mapped_id else item_id)
                    else:

                        item_info = info.copy()
                        if item_mapping and item_id in item_mapping:
                            item_info['mapped_id'] = item_mapping[item_id]
                        items.append(item_info)
        return items

    def get_category_distribution(self) -> Dict[str, int]:
        """Get category distribution statistics"""
        distribution = defaultdict(int)
        for info in self.item_info.values():
            if info and 'categories' in info:
                categories = info['categories']

                if isinstance(categories, list):
                    for category in categories:
                        if isinstance(category, str):
                            distribution[category] += 1
                        elif isinstance(category, list):

                            for sub_cat in category:
                                if isinstance(sub_cat, str):
                                    distribution[sub_cat] += 1
                elif isinstance(categories, str):

                    distribution[categories] += 1
        return dict(distribution)