# -*- coding: utf-8 -*-
# @Time    : 2025/09/17
# @Author  : Claude

import torch
import argparse
import os
import pickle
import numpy as np
from datetime import datetime
from models import SASRec, Linrec, FMLPRecModel, Mamba4Rec, GRU4Rec, TTT4Rec, FilterTTT4Rec, Narm, LightSANs
from utils import set_seed, check_path


class RecommendationGenerator:
    """
    Load trained recommendation model and generate top-K item recommendations for user sequences
    """

    def __init__(self, model_path, model_type, args):
        self.model_path = model_path
        self.model_type = model_type
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.model = None
        self.recommendation_history_file = getattr(args, 'recommendation_history_file', './recommendation_history.txt')
        self.exclude_recommended = getattr(args, 'exclude_recommended', False)
        self.cycle = args.cycle
        self._load_model()

    def _load_recommendation_history(self) -> dict[int, list[int]]:
        """Load user recommendation history"""
        history = {}
        if not os.path.exists(self.recommendation_history_file):
            return history

        try:
            with open(self.recommendation_history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            user_id = int(parts[0])
                            item_ids = [int(item) for item in parts[1:] if item.isdigit()]
                            history[user_id] = item_ids
                        except (ValueError, IndexError):
                            continue

            print(f"✓ Loaded recommendation history for {len(history)} users")
            return history
        except Exception as e:
            print(f"Failed to load recommendation history: {e}")
            return {}

    def _save_recommendation_history(self, history: dict[int, list[int]]):
        """Save user recommendation history"""
        try:

            os.makedirs(os.path.dirname(self.recommendation_history_file), exist_ok=True)

            with open(self.recommendation_history_file, 'w', encoding='utf-8') as f:
                for user_id in sorted(history.keys()):
                    item_ids = history[user_id]
                    if item_ids:
                        line = f"{user_id} " + " ".join(map(str, item_ids)) + "\n"
                        f.write(line)

            print(f"✓ Saved recommendation history for {len(history)} users to: {self.recommendation_history_file}")
        except Exception as e:
            print(f"Failed to save recommendation history: {e}")

    def _load_model(self):
        """Load trained model"""
        model_classes = {
            'SASRec': SASRec,
            'Linrec': Linrec,
            'FMLPRecModel': FMLPRecModel,
            'Mamba4Rec': Mamba4Rec,
            'GRU4Rec': GRU4Rec,
            'TTT4Rec': TTT4Rec,
            'FilterTTT4Rec': FilterTTT4Rec,
            'Narm': Narm,
            'LightSANs': LightSANs
        }

        if self.model_type not in model_classes:
            raise ValueError(f"Model type not found: {self.model_type}")

        self.model = model_classes[self.model_type](self.args)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def predict_for_sequence(self, input_sequence, k=10, exclude_seen=True, exclude_recommended_items=None):
        """
        Generate top-K recommendations for a single user sequence

        Args:
            input_sequence: user interaction sequence, format is list of item IDs
            k: number of recommendations to generate
            exclude_seen: whether to exclude already interacted items
            exclude_recommended_items: list of previously recommended items to exclude

        Returns:
            list of recommended item IDs
        """

        if len(input_sequence) > self.args.max_seq_length:
            input_sequence = input_sequence[-self.args.max_seq_length:]

        pad_len = self.args.max_seq_length - len(input_sequence)
        padded_sequence = [0] * pad_len + input_sequence

        input_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device)

        with torch.no_grad():

            sequence_output = self.model.finetune(input_tensor)

            if self.model_type in ['GRU4Rec', 'Narm']:

                last_hidden = sequence_output[:, -1, :]  # [batch_size, hidden_size]

                scores = torch.matmul(last_hidden, self.model.item_embeddings.weight.t())
            else:

                last_hidden = sequence_output[:, -1, :]  # [batch_size, hidden_size]

                scores = torch.matmul(last_hidden, self.model.item_embeddings.weight.t())

        scores = scores.squeeze(0)  # [item_size]

        scores[0] = float('-inf')

        if exclude_seen:
            seen_items = set(input_sequence)
            for item_id in seen_items:
                if 0 < item_id < len(scores):
                    scores[item_id] = float('-inf')

        if exclude_recommended_items:
            for item_id in exclude_recommended_items:
                if 0 < item_id < len(scores):
                    scores[item_id] = float('-inf')

        _, topk_indices = torch.topk(scores, k, largest=True)

        return topk_indices.cpu().numpy().tolist()

    def predict_batch(self, user_sequences_dict, k=10, exclude_seen=True):
        """
        Batch generate recommendations for multiple user sequences

        Args:
            user_sequences_dict: user sequences dictionary, keys are user IDs, values are interaction sequences
            k: number of recommendations to generate
            exclude_seen: whether to exclude already interacted items

        Returns:
            tuple: (recommendations, probabilities_matrix)
                - recommendations: dictionary, keys are user IDs, values are recommended items
                - probabilities_matrix: numpy 2D array, shape is (num_users, num_items), stores probability for each user-item pair
        """

        recommendation_history = self._load_recommendation_history()

        recommendations = {}
        user_ids = sorted(user_sequences_dict.keys())
        num_users = len(user_ids)
        num_items = self.args.item_size

        probabilities_matrix = np.zeros((num_users, num_items))

        for idx, user_id in enumerate(user_ids):
            sequence = user_sequences_dict[user_id]
            try:

                exclude_recommended_items = None
                if self.exclude_recommended:
                    exclude_recommended_items = recommendation_history.get(user_id, [])

                rec_items = self.predict_for_sequence(sequence, k, exclude_seen, exclude_recommended_items)
                recommendations[user_id] = rec_items

                if user_id in recommendation_history:
                    recommendation_history[user_id].extend(rec_items)
                else:
                    recommendation_history[user_id] = rec_items.copy()

                user_probabilities = self._get_all_item_probabilities(sequence)
                probabilities_matrix[idx] = user_probabilities

                print(f"✓ User {user_id} recommendation complete: {len(rec_items)} items")
                if self.exclude_recommended and exclude_recommended_items:
                    print(f"  Excluded {len(exclude_recommended_items)} previously recommended items")

            except Exception as e:
                print(f"✗ User {user_id} recommendation failed: {e}")
                recommendations[user_id] = []

        if self.exclude_recommended:
            self._save_recommendation_history(recommendation_history)

        return recommendations, probabilities_matrix

    def _get_all_item_probabilities(self, input_sequence):
        """
        Get probability scores for all items for a user

        Args:
            input_sequence: user interaction sequence, format is list of item IDs

        Returns:
            numpy array containing probability scores for all items for this user
        """

        if len(input_sequence) > self.args.max_seq_length:
            input_sequence = input_sequence[-self.args.max_seq_length:]

        pad_len = self.args.max_seq_length - len(input_sequence)
        padded_sequence = [0] * pad_len + input_sequence

        input_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device)

        with torch.no_grad():

            sequence_output = self.model.finetune(input_tensor)

            if self.model_type in ['GRU4Rec', 'Narm']:

                last_hidden = sequence_output[:, -1, :]  # [batch_size, hidden_size]

                scores = torch.matmul(last_hidden, self.model.item_embeddings.weight.t())
            else:

                last_hidden = sequence_output[:, -1, :]  # [batch_size, hidden_size]

                scores = torch.matmul(last_hidden, self.model.item_embeddings.weight.t())

        scores = scores.squeeze(0)  # [item_size]

        probabilities = torch.softmax(scores, dim=0)

        return probabilities.cpu().numpy()

    def save_recommendations(self, recommendations, output_path):
        """
        Save recommendation results to file, and create backup with timestamp

        Args:
            recommendations: recommendation results dictionary
            output_path: output file path
        """
        check_path(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            for user_id in sorted(recommendations.keys()):
                rec_items = recommendations[user_id]
                if rec_items:
                    line = f"{user_id} " + " ".join(map(str, rec_items)) + "\n"
                    f.write(line)

        print(f"✓ Recommendation results saved to: {output_path}")

        backup_path = self._add_cycle_to_filename(output_path)

        with open(backup_path, 'w') as f:
            for user_id in sorted(recommendations.keys()):
                rec_items = recommendations[user_id]
                if rec_items:
                    line = f"{user_id} " + " ".join(map(str, rec_items)) + "\n"
                    f.write(line)

        print(f"✓ Recommendation results saved to: {backup_path}")

    def save_probabilities_matrix(self, probabilities_matrix, user_ids, output_dir):
        """
        Save user-item probability matrix to numpy file

        Args:
            probabilities_matrix: numpy 2D array, shape is (num_users, num_items)
            user_ids: user ID list, corresponding to matrix rows
            output_dir: output directory
        """
        check_path(output_dir)

        prob_filename = f"user_item_probabilities_{self.cycle}.npy"
        user_ids_filename = f"user_ids_{self.cycle}.npy"

        prob_path = os.path.join(output_dir, prob_filename)
        user_ids_path = os.path.join(output_dir, user_ids_filename)

        np.save(prob_path, probabilities_matrix)
        print(f"✓ User-item probabilities saved to: {prob_path}")
        print(f"  Probability matrix shape: {probabilities_matrix.shape}")

        np.save(user_ids_path, np.array(user_ids))
        print(f"✓ User ID mapping saved to: {user_ids_path}")

    def _add_cycle_to_filename(self, file_path):
        """
        Add timestamp to filename

        Args:
            file_path: original file path

        Returns:
            new file path with timestamp
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        new_filename = f"{name}_{self.cycle}{ext}"
        return os.path.join(dir_name, new_filename)


def load_user_sequences(data_path, format_type='pickle'):
    """
    Load user sequence data

    Args:
        data_path: data file path
        format_type: data format ('pickle', 'txt', 'json')

    Returns:
        user sequences dictionary, keys are user IDs, values are interaction sequences
    """
    user_sequences = {}
    item_set = set()
    if format_type == 'pickle':
        with open(data_path, 'rb') as f:
            sequences = pickle.load(f)

            for i, seq in enumerate(sequences, 1):
                user_sequences[i] = seq
    elif format_type == 'txt':
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    items = list(map(int, line.strip().split()))
                    if len(items) > 0:
                        user_id = items[0]
                        sequence = items[1:]
                        item_set = item_set | set(sequence)
                        user_sequences[user_id] = sequence
    else:
        raise ValueError(f"Unsupported data format: {format_type}")
    max_item = max(item_set)
    return user_sequences, max_item


def generate_user_range_sequences(start_user, end_user, data_path, format_type='txt'):
    """
    Generate sequence data for specified user range

    Args:
        start_user: starting user ID
        end_user: ending user ID
        data_path: data file path
        format_type: data format

    Returns:
        user sequences dictionary, keys are user IDs, values are interaction sequences
    """
    all_sequences, max_item = load_user_sequences(data_path, format_type)

    user_sequences = {}
    for user_id in range(start_user, end_user + 1):
        if user_id in all_sequences:
            user_sequences[user_id] = all_sequences[user_id]
        else:

            print(f"Warning: No data found for user {user_id}, skipping")

    return user_sequences, max_item


def create_args_from_config(config_dict):
    """Create args object from config dictionary"""
    class Args:
        pass

    args = Args()
    for key, value in config_dict.items():
        setattr(args, key, value)

    return args


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['SASRec', 'Linrec', 'FMLPRecModel', 'Mamba4Rec', 'GRU4Rec', 'TTT4Rec', 'FilterTTT4Rec', 'Narm', 'LightSANs'],
                        help='Type of model to use')

    parser.add_argument('--data_path', type=str, required=True, help='Path to user sequence data file')
    parser.add_argument('--data_format', type=str, default='txt', choices=['pickle', 'txt'], help='Format of data file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save recommendation results')
    parser.add_argument('--prob_output_dir', type=str, default=None, help='Directory to save probability matrices (optional)')

    parser.add_argument('--start_user', type=int, default=1, help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=100, help='Ending user ID')

    parser.add_argument('--k', type=int, default=10, help='Number of recommendations to generate')
    parser.add_argument('--exclude_seen', action='store_true', default=True, help='Exclude already interacted items')
    parser.add_argument('--exclude_recommended', default=False, help='Exclude previously recommended items')
    parser.add_argument('--recommendation_history_file', type=str, default='./recommendation_history.txt', help='Path to recommendation history file')

    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--max_seq_length', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5, help='Dropout probability for hidden layers')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2, help='Dropout probability for attention')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='Initializer range')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='Hidden layer activation function')
    parser.add_argument('--cycle', type=int, default=1, help='Current cycle number')

    parser.add_argument('--d_state', type=int, default=16, help='D state for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='D conv for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='Expand for Mamba')

    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    args.cuda_condition = True

    set_seed(args.seed)

    print(f"Running: Loading users {args.start_user}-{args.end_user} data from: {args.data_path}")
    user_sequences, max_item = generate_user_range_sequences(args.start_user, args.end_user, args.data_path, args.data_format)
    args.item_size = max_item + 2
    print(f"✓ Loading complete, {len(user_sequences)} users found")

    if len(user_sequences) == 0:
        print("Error: No user sequence data found")
        return

    print(f"\nRunning: Loading model: {args.model_type}")
    recommender = RecommendationGenerator(args.model_path, args.model_type, args)

    print(f"\nRunning: Generating Top-{args.k} recommendations for users {args.start_user}-{args.end_user}...")
    if hasattr(args, 'exclude_recommended') and args.exclude_recommended:
        print(f"  Using recommendation history file: {args.recommendation_history_file}")
        print(f"  Excluding previously recommended items")
    recommendations, probabilities_matrix = recommender.predict_batch(user_sequences, args.k, args.exclude_seen)

    recommender.save_recommendations(recommendations, args.output_path)

    if args.prob_output_dir:
        os.makedirs(args.prob_output_dir, exist_ok=True)
        user_ids = sorted(user_sequences.keys())
        recommender.save_probabilities_matrix(probabilities_matrix, user_ids, args.prob_output_dir)

    print("\n🎉 Recommendation generation complete!")


if __name__ == '__main__':
    main()
