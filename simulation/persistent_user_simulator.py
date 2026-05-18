#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent User Simulator - Closed-loop interaction between recommendation system and Agent users

Features:
- Users persist after creation, retaining memory state
- Supports multi-round cycle simulation without recreating users
- Integrates recommendation models from recommenders
- Automatically saves interaction data and updates training dataset
- Supports evolution of user profiles and behavior patterns
- Reuses LLMAvatar model implemented in avatar.py as underlying user simulator

Implementation notes:
- Uses LLMAvatar class from avatar.py as core for user decisions
- Adds persistence layer to support saving/loading user profiles and interaction history
- Provides user manager for unified management of multiple users
- Maintains compatibility with existing avatar.py interface
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from avatar import LLMAvatar, create_avatar

from dotenv import load_dotenv
load_dotenv()


@dataclass
class InteractionRecord:
    """Interaction record data class"""
    user_id: str
    session_id: str
    recommended_items: List[Dict[str, Any]]
    selected_items: List[Dict[str, Any]]
    timestamp: float
    cycle_number: int
    user_feedback: Dict[str, Any]


class PersistentUserSimulator:
    """Persistent user simulator, only uses memory to save interaction memory"""

    def __init__(self,
                 user_id: str,
                 persona_text: str,
                 memory_size: int = 5,
                 temperature: float = 0.3,
                 model_name: str = "gpt-5-mini",
                 memory_storage_dir: str = "./user_memory",
                 **llm_kwargs):
        """
        Initialize persistent user simulator

        Args:
            user_id: User ID
            persona_text: User persona text
            memory_size: Memory window size
            temperature: LLM temperature parameter
            model_name: Model name to use
            storage_dir: User data storage directory (deprecated, only kept for compatibility)
            memory_storage_dir: Memory file storage directory
            **llm_kwargs: Other LLM parameters
        """
        self.user_id = user_id
        self.persona_text = persona_text
        self.memory_size = memory_size
        self.memory_storage_dir = memory_storage_dir

        self.avatar = create_avatar(
            user_id=user_id,
            persona_text=persona_text,
            memory_size=memory_size,
            temperature=temperature,
            model_name=model_name,
            memory_storage_dir=memory_storage_dir,
            **llm_kwargs
        )

        self.total_interactions = 0
        self.favorite_categories = []
        self.created_at = time.time()
        self.last_active = time.time()
        self.current_session_id = None
        self.cycle_number = 0



    def start_new_session(self, cycle_number: int):
        """Start new interaction session"""
        self.current_session_id = f"session_{int(time.time())}_{cycle_number}"
        self.cycle_number = cycle_number
        print(f"User {self.user_id} starting Round {cycle_number} session: {self.current_session_id}")

        self.avatar.start_new_round()

    def evaluate_recommendations(self, recommended_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate recommended items list and save detailed interaction memory

        Args:
            recommended_items: List of recommended items

        Returns:
            List of evaluation results
        """
        if not self.current_session_id:
            raise ValueError("Please call start_new_session() first to start a session")

        evaluations = []
        selected_items = []

        print(f"\n--- User {self.user_id} evaluating {len(recommended_items)} recommended items ---")

        try:
            decision = self.avatar.decide_items(recommended_items, cycle=self.cycle_number)
            selected_item_id = decision.get("selected_item_id")

            if selected_item_id is not None:
                selected_item = None
                for item in recommended_items:
                    if item.get('item_id') == selected_item_id:
                        selected_item = item.copy()
                        selected_item.update({
                            'reason': decision.get('reason', ''),
                        })
                        selected_items.append(selected_item)
                        break

                if selected_item:
                    print(f"✓ User selected item {selected_item_id}: {decision.get('reason', '')}")
                else:
                    print(f"⚠ User selected item ID {selected_item_id} not found in recommendations")
            else:
                print(f"✗ User did not select any item: {decision.get('reason', 'No reason provided')}")

            for item in recommended_items:
                item_id = item.get('item_id', item.get('id'))
                if item_id == selected_item_id:
                    evaluations.append({
                        "item_id": item_id,
                        "interact": True,
                        "reason": decision.get('reason', ''),
                        "selected_item_id": selected_item_id
                    })
                else:
                    evaluations.append({
                        "item_id": item_id,
                        "interact": False,
                        "reason": "Not selected",
                        "selected_item_id": None
                    })

        except Exception as e:
            print(f"Evaluation failed: {e}")

            for i, item in enumerate(recommended_items):
                decision = {
                    "item_id": item.get('item_id', item.get('id', str(i))),
                    "interact": False,
                    "reason": f"Decision failed: {str(e)}",
                    "error": True,
                    "selected_item_id": None
                }
                evaluations.append(decision)

        # self._save_interaction_memory(recommended_items, selected_items, evaluations)

        self.total_interactions += len(selected_items)
        self.last_active = time.time()

        self.avatar.save_memory()

        response_stats = self.avatar.get_current_round_response_stats()
        if response_stats["count"] > 0:
            print(f"\n--- User {self.user_id} Round {self.cycle_number} LLM API stats ---")
            print(f"API calls: {response_stats['count']}")
            print(f"Total time: {response_stats['total_time']:.3f}s")
            print(f"Average time: {response_stats['avg_time']:.3f}s")
            print(f"Minimum time: {response_stats['min_time']:.3f}s")
            print(f"Maximum time: {response_stats['max_time']:.3f}s")
            print("-" * 60)

        return evaluations


    def _update_preference_stats(self, selected_items: List[Dict[str, Any]]):
        """Update preference statistics (memory only)"""
        preference_counts = {}
        
        for item in selected_items:
            category = item.get('first_level_category_name', item.get('category', 'unknown'))
            if category != 'unknown':
                preference_counts[category] = preference_counts.get(category, 0) + 1

        if preference_counts:
            for category, count in preference_counts.items():
                if category not in self.favorite_categories:
                    self.favorite_categories.append(category)
            
            self.favorite_categories = self.favorite_categories[:5]

    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""

        total_response_stats = self.avatar.get_total_response_stats()

        return {
            "user_id": self.user_id,
            "total_interactions": self.total_interactions,
            "cycles_participated": self.cycle_number,
            "favorite_categories": self.favorite_categories,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "api_response_stats": total_response_stats
        }

    def reset_memory(self):
        """Reset memory (preserve user persona)"""
        self.avatar.reset_memory()

    def export_interaction_data(self, format_type: str = "txt") -> str:
        """
        Export user interaction data for training
        Note: Due to memory-only mode, only interaction history in current memory can be exported

        Args:
            format_type: Export format ("txt" or "json")

        Returns:
            Exported data string
        """

        try:
            memory_data = self.avatar.get_memory_summary()
            if format_type == "txt":
                return f"# User {self.user_id} interaction data from memory\n# Total interactions: {self.total_interactions}\n# Current memory size: {self.memory_size}\n"
            else:
                return json.dumps({
                    "user_id": self.user_id,
                    "total_interactions": self.total_interactions,
                    "memory_summary": memory_data,
                    "favorite_categories": self.favorite_categories
                }, indent=2)
        except Exception as e:
            print(f"Data export failed: {e}")
            return ""


class UserManager:
    """User manager, responsible for creating and managing multiple persistent users"""

    def __init__(self,  memory_storage_dir: str = None):
        if memory_storage_dir is None :
            raise Exception("memory_storage_dir is None")
            
        self.memory_storage_dir = memory_storage_dir
        self.users: Dict[str, PersistentUserSimulator] = {}
        os.makedirs(memory_storage_dir, exist_ok=True)

    def create_or_load_user(self, user_id: str, persona_text: str, **kwargs) -> PersistentUserSimulator:
        """Create or load user"""
        if user_id in self.users:
            return self.users[user_id]

        kwargs['memory_storage_dir'] = self.memory_storage_dir
        user = PersistentUserSimulator(user_id, persona_text, **kwargs)
        self.users[user_id] = user
        return user

    def get_all_users(self) -> List[PersistentUserSimulator]:
        """Get all users"""
        return list(self.users.values())

    def get_user(self, user_id: str) -> Optional[PersistentUserSimulator]:
        """Get user by user ID"""
        return self.users.get(user_id)

    def save_all_users(self):
        """Save all user data - No manual saving needed in memory-only mode"""
        print(f"Memory-only mode: {len(self.users)} users not saved to file")

    def get_user_stats_summary(self) -> Dict[str, Any]:
        """Get user statistics summary"""
        if not self.users:
            return {}

        all_stats = [user.get_user_stats() for user in self.users.values()]

        return {
            "total_users": len(all_stats),
            "total_interactions": sum(stat["total_interactions"] for stat in all_stats),
            "avg_interactions_per_user": np.mean([stat["total_interactions"] for stat in all_stats]),
            "max_cycles": max(stat["cycles_participated"] for stat in all_stats),
            "user_details": all_stats
        }

    def export_all_interaction_data(self, output_file: str, format_type: str = "txt"):
        """Export all users' interaction data"""
        all_data = []
        for user in self.users.values():
            user_data = user.export_interaction_data(format_type)
            if user_data.strip():
                all_data.append(user_data)

        if format_type == "txt":
            combined_data = "\n".join(all_data)
        else:
            combined_data = "[\n" + ",\n".join(all_data.split("\n")) + "\n]"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_data)

            print(f"✓ Exported {len(self.users)} users' data to {output_file}")


def main():
    """Main function for testing"""
    parser = argparse.ArgumentParser(description="Persistent user simulator test")
    parser.add_argument('--user_id', type=str, default='test_user')
    parser.add_argument('--persona_file', type=str, default='../datasets/persona_description/user_0_profile.txt')
    parser.add_argument('--memory_storage_dir', type=str, default='./user_memory')

    args = parser.parse_args()

    if os.path.exists(args.persona_file):
        with open(args.persona_file, 'r', encoding='utf-8') as f:
            persona_text = f.read().strip()
    else:
        persona_text = "I am a user who enjoys various types of content and likes to explore new recommendations."

    manager = UserManager(args.memory_storage_dir)

    user = manager.create_or_load_user(args.user_id, persona_text)

    print("User stats:", user.get_user_stats())
    print(f"✓ User creation completed")
    print(f"✓ Memory stored at: {args.memory_storage_dir}")


if __name__ == '__main__':
    main()
