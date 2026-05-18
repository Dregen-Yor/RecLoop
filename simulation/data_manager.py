#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Manager - Handles interaction data persistence and dataset updates

Features:
- Automatically save new interaction data to training datasets
- Support incremental data updates and merging
- Data format conversion and validation
- Dataset backup and recovery
- Support integration of multiple data sources
"""

import os
import json
import time
import shutil
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

from persistent_user_simulator import UserManager, PersistentUserSimulator


class InteractionDataManager:
    """
    Interaction data manager, responsible for data persistence, merging, and updates
    """

    def __init__(self,
                 dataset_name: str = "Toys_and_Games",
                 storage_dir: str = "./simulation_storage",
                 backup_enabled: bool = True):
        """
        Initialize data manager

        Args:
            dataset_name: Dataset name
            storage_dir: Simulation data storage directory
            backup_enabled: Whether to enable automatic backup
        """
        self.dataset_name = dataset_name
        self.storage_dir = Path(storage_dir)
        self.backup_enabled = backup_enabled

        self.data_file = Path(str(self.storage_dir) + "/data/"+dataset_name+"-0.txt")

        self.interactions_dir = self.storage_dir / "interactions"
        self.incremental_dir = self.storage_dir / "incremental_data"
        self.backup_dir = self.storage_dir / "backups"

        for dir_path in [self.interactions_dir, self.incremental_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

 print(f"✓ Data manager initialized - dataset: {dataset_name}")

    def save_interaction_batch(self,
                              interactions: List[Dict[str, Any]],
                              batch_id: str,
                              cycle_number: int) -> str:
        """
        Save a batch of interaction data

        Args:
            interactions: List of interaction data
            batch_id: Batch ID
            cycle_number: Cycle number

        Returns:
            Saved file path
        """
        timestamp = int(time.time())
        filename = f"interactions_cycle_{cycle_number}_batch_{batch_id}_{timestamp}.json"
        filepath = self.interactions_dir / filename

        batch_data = {
            "batch_id": batch_id,
            "cycle_number": cycle_number,
            "timestamp": timestamp,
            "total_interactions": len(interactions),
            "data": interactions
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

 print(f"✓ Saved {len(interactions)} interactions to {filepath}")
        return str(filepath)

    def merge_interactions_to_training_data(self,
                                          interaction_files: Optional[List[str]] = None,
                                          min_interactions_per_user: int = 1) -> Dict[str, Any]:
        """
        Merge interaction data into training dataset

        Args:
            interaction_files: List of interaction data files, auto-find if None
            min_interactions_per_user: Minimum interactions per user

        Returns:
            Merge statistics
        """

        if interaction_files is None:
            interaction_files = self._find_recent_interaction_files()

        if not interaction_files:
 print("No interaction data files found")
            return {"status": "no_data"}

        if self.backup_enabled:
            self._backup_data_file()

        existing_data = self._load_training_data()

        merged_data = self._merge_interaction_data(existing_data, interaction_files)

        cleaned_data = self._clean_training_data(merged_data, min_interactions_per_user)

        stats = self._save_updated_training_data(cleaned_data)

 print(f"✓ Data merge completed: {stats}")
        return stats

    def _find_recent_interaction_files(self, hours: int = 24) -> List[str]:
        """Find recent interaction data files"""
        cutoff_time = time.time() - (hours * 3600)
        files = []

        for file_path in self.interactions_dir.glob("*.json"):
            if file_path.stat().st_mtime > cutoff_time:
                files.append(str(file_path))

        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files

    def _backup_data_file(self):
        """Backup data file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_name = f"data_backup_{timestamp}.txt"

        if self.data_file.exists():
            backup_path = self.backup_dir / backup_name
            shutil.copy2(self.data_file, backup_path)
 print(f"✓ Backed up data file to {backup_path}")

    def _load_training_data(self) -> Dict[int, List[int]]:
        """Load existing data (unified data file)"""
        user_sequences = {}

        if not self.data_file.exists():
 print("Data file does not exist, creating empty dataset")
            return user_sequences

        with open(self.data_file, 'r', encoding='utf-8') as f:
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
                        user_sequences[user_id] = item_ids
                except (ValueError, IndexError) as e:
 print(f"Data parsing failed: {line} - {e}")
                    continue

 print(f"✓ Loaded {len(user_sequences)} users from training data")
        return user_sequences

    def _merge_interaction_data(self,
                               existing_data: Dict[int, List[int]],
                               interaction_files: List[str]) -> Dict[int, List[int]]:
        """Merge interaction data into existing data"""
        merged_data = existing_data.copy()

        total_new_interactions = 0

        for file_path in interaction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)

                interactions = batch_data.get('data', [])

                for interaction in interactions:
                    user_id = interaction.get('user_id')
                    if not isinstance(user_id, int):
                        continue

                    new_items = self._extract_items_from_interaction(interaction)

                    if not new_items:
                        continue

                    if user_id not in merged_data:
                        merged_data[user_id] = []

                    merged_data[user_id].extend(new_items)
                    total_new_interactions += len(new_items)

            except Exception as e:
 print(f"Processing file failed {file_path}: {e}")
                continue

 print(f"✓ Added {total_new_interactions} new interactions")
        return merged_data

    def _extract_items_from_interaction(self, interaction: Dict[str, Any]) -> List[int]:
        """Extract item IDs from interaction record"""
        items = []

        watched_items = interaction.get('watched_items', [])
        for item in watched_items:
            if isinstance(item, dict):
                item_id = item.get('video_id') or item.get('item_id')
            else:
                item_id = item

            if item_id is not None:
                try:
                    items.append(int(item_id))
                except (ValueError, TypeError):
                    continue

        return items

    def _clean_training_data(self,
                           data: Dict[int, List[int]],
                           min_interactions: int) -> Dict[int, List[int]]:
        """Clean training data (preserve duplicates)"""
        cleaned_data = {}

        for user_id, items in data.items():
            if len(items) >= min_interactions:
                cleaned_data[user_id] = items

 print(f"✓ Data cleaning completed: {len(data)} -> {len(cleaned_data)} users")
        return cleaned_data

    def _save_updated_training_data(self, data: Dict[int, List[int]]) -> Dict[str, Any]:
        """Save updated data file"""

        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, 'w', encoding='utf-8') as f:
            for user_id in sorted(data.keys()):
                items = data[user_id]
                if items:
                    line = f"{user_id} {' '.join(map(str, items))}\n"
                    f.write(line)

        total_users = len(data)
        total_interactions = sum(len(items) for items in data.values())
        avg_interactions = total_interactions / total_users if total_users > 0 else 0

        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "avg_interactions_per_user": round(avg_interactions, 2),
            "file_path": str(self.data_file)
        }

 print(f"✓ Saved updated data file: {stats}")
        return stats

    def export_user_interaction_summary(self,
                                      output_file: str,
                                      format_type: str = "json") -> str:
        """
        Export user interaction summary

        Args:
            output_file: Output file path
            format_type: Output format ("json" or "csv")

        Returns:
            Output file path
        """

        training_data = self._load_training_data()

        if format_type == "json":

            summary = {
                "dataset": self.dataset_name,
                "total_users": len(training_data),
                "total_interactions": sum(len(items) for items in training_data.values()),
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "user_summary": []
            }

            for user_id, items in training_data.items():
                user_summary = {
                    "user_id": user_id,
                    "interaction_count": len(items),
                    "items": items,
                    "categories": self._get_user_categories(items)
                }
                summary["user_summary"].append(user_summary)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        else:

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("user_id,interaction_count,items\n")
                for user_id, items in training_data.items():
                    items_str = ",".join(map(str, items))
                    f.write(f"{user_id},{len(items)},{items_str}\n")

 print(f"✓ Exported user interaction summary to {output_file}")
        return output_file

    def _get_user_categories(self, items: List[int]) -> List[str]:
        """Get user's item category distribution (simplified implementation)"""
        return ["mixed"] * min(3, len(items))

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        training_data = self._load_training_data()

        if not training_data:
            return {"status": "empty"}

        interaction_counts = [len(items) for items in training_data.values()]
        unique_items = set()
        for items in training_data.values():
            unique_items.update(items)

        stats = {
            "dataset": self.dataset_name,
            "total_users": len(training_data),
            "total_interactions": sum(interaction_counts),
            "unique_items": len(unique_items),
            "avg_interactions_per_user": round(np.mean(interaction_counts), 2),
            "median_interactions_per_user": int(np.median(interaction_counts)),
            "max_interactions_per_user": max(interaction_counts),
            "min_interactions_per_user": min(interaction_counts),
            "user_distribution": self._get_user_distribution(interaction_counts)
        }

        return stats

    def _get_user_distribution(self, interaction_counts: List[int]) -> Dict[str, int]:
        """Get user interaction distribution"""
        bins = [0, 5, 10, 20, 50, 100, float('inf')]
        labels = ["0-5", "6-10", "11-20", "21-50", "51-100", "100+"]

        distribution = {}
        for i, label in enumerate(labels):
            if i < len(bins) - 1:
                count = sum(1 for count in interaction_counts
                          if bins[i] <= count < bins[i+1])
            else:
                count = sum(1 for count in interaction_counts
                          if count >= bins[i])
            distribution[label] = count

        return distribution

    def cleanup_old_files(self, days_to_keep: int = 7):
        """Clean up old interaction data files"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        cleaned_files = 0
        for file_path in self.interactions_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                cleaned_files += 1

        if cleaned_files > 0:
 print(f"✓ Cleaned up {cleaned_files} old data files")

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity"""
        issues = []

        if not self.data_file.exists():
            issues.append("Data file does not exist")

        training_data = self._load_training_data()
        if training_data:
            user_ids = list(training_data.keys())
            if user_ids:
                min_id, max_id = min(user_ids), max(user_ids)
                missing_ids = set(range(min_id, max_id + 1)) - set(user_ids)
                if missing_ids:
                    issues.append(f"User IDs are not continuous, missing: {len(missing_ids)} IDs")

            for user_id, items in training_data.items():
                invalid_items = [item for item in items if not isinstance(item, int) or item < 0]
                if invalid_items:
                    issues.append(f"User {user_id} has invalid item IDs: {invalid_items}")

        result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "checked_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return result


class IncrementalTrainingDataGenerator:
    """
    Incremental training data generator, generates new training data from user simulator
    """

    def __init__(
                 self,
                 user_manager: UserManager,
                 data_manager: InteractionDataManager):
        """
        Initialize incremental data generator

        Args:
            user_manager: User manager
            data_manager: Data manager
        """
        self.user_manager = user_manager
        self.data_manager = data_manager

    def generate_incremental_data(self,
                                cycle_number: int,
                                min_interactions_threshold: int = 1) -> Dict[str, Any]:
        """
        Generate incremental training data

        Args:
            cycle_number: Current cycle number
            min_interactions_threshold: Minimum interactions threshold

        Returns:
            Generation statistics
        """
 print(f"\n=== Generating round {cycle_number} training data ===")

        try:

            all_interactions = []
            total_new_interactions = 0

            users = self.user_manager.get_all_users()
            for user in users:

                user_data = user.export_interaction_data("json")
                if user_data.strip():
                    try:
                        user_info = json.loads(user_data)
                        if user_info and isinstance(user_info, dict):

                            user_interaction_record = {
                                "user_id": user_info.get("user_id", user.user_id),
                                "cycle_number": cycle_number,
                                "total_interactions": user_info.get("total_interactions", 0),
                                "memory_summary": user_info.get("memory_summary", []),
                                "favorite_categories": user_info.get("favorite_categories", []),
                                "timestamp": time.time()
                            }
                            all_interactions.append(user_interaction_record)
                            

                            total_new_interactions += user_info.get("total_interactions", 0)
                    except json.JSONDecodeError as e:
 print(f"User {user.user_id} data parsing failed: {e}")
                        continue

            if not all_interactions:
 print("No new interaction data to process")
                return {"status": "no_new_data"}


            batch_id = f"cycle_{cycle_number}"
            saved_file = self.data_manager.save_interaction_batch(
                all_interactions, batch_id, cycle_number
            )


            merge_stats = self.data_manager.merge_interactions_to_training_data(
                min_interactions_per_user=min_interactions_threshold
            )

            stats = {
                "cycle_number": cycle_number,
                "total_users": len(users),
                "new_interactions": total_new_interactions,
                "saved_file": saved_file,
                "merge_stats": merge_stats,
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }

 print(f"✓ Data generation completed: {stats}")
            return stats
            
        except Exception as e:
 print(f"Data generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "cycle_number": cycle_number
            }

    def get_incremental_data_summary(self) -> Dict[str, Any]:
        """Get incremental data summary"""

        data_stats = self.data_manager.get_data_statistics()

        user_stats = self.user_manager.get_user_stats_summary()

        summary = {
            "data_stats": data_stats,
            "user_stats": user_stats,
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return summary


def main():
    """Main function for testing"""
    parser = argparse.ArgumentParser(description="Data manager test")
 parser.add_argument('--dataset', type=str, default='KuaiRec', help='Dataset name')
    parser.add_argument('--action', type=str, choices=['stats', 'merge', 'export', 'validate'],
 default='stats', help='Action to perform')
 parser.add_argument('--output_file', type=str, help='Output file path')

    args = parser.parse_args()


    data_manager = InteractionDataManager(dataset_name=args.dataset)

    if args.action == 'stats':

        stats = data_manager.get_data_statistics()
 print("Data statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.action == 'merge':

        stats = data_manager.merge_interactions_to_training_data()
 print("Merge statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.action == 'export' and args.output_file:

        data_manager.export_user_interaction_summary(args.output_file)

    elif args.action == 'validate':

        result = data_manager.validate_data_integrity()
 print("Data validation result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
 print("Invalid action specified")


if __name__ == '__main__':
    main()
