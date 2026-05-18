"""
User Profile Extractor
Extract user profiles from user rating records by providing prompts to LLM
"""

import os
import json
import logging
import asyncio
import re
import openai
import numpy as np
import random
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import dotenv
from tqdm import tqdm
from item_info_retriever import ItemInfoRetriever
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import APIClient


dotenv.load_dotenv()

class UserProfileExtractor:
    def __init__(self,
                 dataset_name: str = "Beauty_and_Personal_Care",
                 api_config_path: str = "api_config.json",
                 save_prompts: bool = True,
                 prompts_dir: str = "user_prompts"):
        """
        Initialize user profile extractor

        Args:
            dataset_name: Dataset name
            api_config_path: API configuration file path
            save_prompts: Whether to save generated prompts to local files
            prompts_dir: Directory name for saving prompts
        """

        self.dataset_path =  f"./recommenders/data/{dataset_name}"

        self.dataset_name = dataset_name
        self.save_prompts = save_prompts
        self.prompts_dir = prompts_dir 


        self.api_client = APIClient(api_config_path)
        

        self.item_retriever = ItemInfoRetriever(self.dataset_path, dataset_name)


        self.executor = ThreadPoolExecutor(max_workers=30, thread_name_prefix="profile_extractor")


        self.interaction_file = os.path.join(self.dataset_path, f"{dataset_name}.txt")
        self.review_file = os.path.join(self.dataset_path, f"{dataset_name}.json")
        self.user_mapping_file = os.path.join(self.dataset_path, 'user_mapping.npy')
        self.item_mapping_file = os.path.join(self.dataset_path, 'item_mapping.npy')


        self._setup_logging()


        self.user_mapping = {}
        self.reverse_user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self._load_user_mappings()
        self._load_item_mappings()


        self.interaction_data = self._load_interaction_data()
        self.review_data = self._load_review_data()

    def __del__(self):
        """Clean up thread pool resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_user_mappings(self):
        """Load user mapping file, similar to item_info_retriever.py"""
        self.logger.info("Starting to load user mapping file")

        if os.path.exists(self.user_mapping_file):
            try:
                self.user_mapping = np.load(self.user_mapping_file, allow_pickle=True).item()
                self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}

                return
            except Exception as e:
                self.logger.warning(f"Failed to load user mapping file {self.user_mapping_file}: {e}")

        self.logger.warning("No valid user mapping file found, using original user IDs")

    def _load_item_mappings(self):
        """Load item mapping file"""
        self.logger.info("Starting to load item mapping file")

        if os.path.exists(self.item_mapping_file):
            try:
                self.item_mapping = np.load(self.item_mapping_file, allow_pickle=True).item()
                self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
                self.logger.info(f"Loaded item mapping file: {self.item_mapping_file}")
                self.logger.info(f"Item mapping size: {len(self.item_mapping)}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load item mapping file {self.item_mapping_file}: {e}")

        self.logger.warning("No valid item mapping file found, using original IDs")
        
    def _resolve_user_id(self, user_id) -> Optional[str]:
        """Resolve user ID, return original user ID string
        
        Args:
            user_id: Input user ID (may be integer or string)
            
        Returns:
            Original user ID string or None
        """

        if isinstance(user_id, str):

            if user_id in self.review_data:
                return user_id
            

            try:
                user_id = int(user_id)
            except ValueError:

                return None
        

        if isinstance(user_id, int) and self.reverse_user_mapping:
            original_id = self.reverse_user_mapping.get(user_id)
            if original_id:
                return original_id
        
        return None
        
    def _load_interaction_data(self) -> Dict[str, List[str]]:
        """Load simplified interaction data"""
        interaction_data = {}
        try:
            with open(self.interaction_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        user_id = parts[0]
                        items = parts[1:]
                        interaction_data[user_id] = items
                self.logger.info(f"Loaded data completed, User count: {len(interaction_data)}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            interaction_data = {}
        return interaction_data
    
    def _load_review_data(self) -> Dict[str, List[Dict]]:
        """Load complete review information data"""
        review_data = defaultdict(list)
        try:
            with open(self.review_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        review = json.loads(line.strip())
                        user_id = review.get('reviewerID')
                        if user_id:
                            review_data[user_id].append(review)
                    except json.JSONDecodeError as e:
                        if line_num < 10:
                            self.logger.warning(f"Line {line_num} failed: {e}")
                        continue
                self.logger.info(f"Loaded data completed, User count: {len(review_data)}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            review_data = defaultdict(list)
        return dict(review_data)
    
    def get_user_purchase_history(self, user_id) -> List[Dict]:
        """Get user's purchase history, filter items from interaction_data in review_data"""
        history = []

        original_user_id = self._resolve_user_id(user_id)
        if not original_user_id:
            self.logger.warning(f"User ID not found: {user_id}")
            return history


        interacted_item_ids = set()


        lookup_keys = [
            str(user_id),
            original_user_id if isinstance(original_user_id, str) else str(original_user_id),
            original_user_id
        ]


        if isinstance(user_id, str):
            try:
                numeric_user_id = int(user_id)
                lookup_keys.append(str(numeric_user_id))
            except ValueError:
                pass

        for key in lookup_keys:
            if key in self.interaction_data:
                interacted_item_ids = set(self.interaction_data[key])
                self.logger.debug(f"foundusers {user_id} data, : {key}")
                break

        if not interacted_item_ids:
            self.logger.warning(f"users {user_id} data")
            return history


        original_item_ids = set()
        for item_id_str in interacted_item_ids:

            try:
                item_id = int(item_id_str)
            except ValueError:
                self.logger.warning(f"ID: {item_id_str}")
                continue


            if self.reverse_item_mapping and item_id in self.reverse_item_mapping:
                original_asin = self.reverse_item_mapping[item_id]
                original_item_ids.add(original_asin)
            else:
                self.logger.warning(f"ID {item_id} foundmapping")

        self.logger.debug(f"users {user_id} {len(interacted_item_ids)} mappingID, {len(original_item_ids)} ID")

        reviews = self.review_data.get(original_user_id, [])


        review_by_item = {}
        matched_reviews = 0

        for review in reviews:
            asin = review.get('asin')
            if asin and asin in original_item_ids:
                review_by_item[asin] = review
                matched_reviews += 1

        self.logger.info(f"users {user_id} data {len(original_item_ids)} , data {len(reviews)} , {matched_reviews} ")

        for original_asin in original_item_ids:

            item_info = self.item_retriever.get_item_info(original_asin)


            review = review_by_item.get(original_asin)

            if review:

                purchase_record = {
                    'asin': original_asin,
                    'rating': review.get('overall'),
                    'title': review.get('title', ''),
                    'review_text': review.get('reviewText', ''),
                    'timestamp': review.get('unixReviewTime'),
                    'item_title': item_info.get('title', '') if item_info else '',
                    'item_categories': item_info.get('categories', []) if item_info else [],
                    'item_description': item_info.get('description', '') if item_info else ''
                }
            else:

                purchase_record = {
                    'asin': original_asin,
                    'rating': None,
                    'title': '',
                    'review_text': '',
                    'timestamp': None,
                    'item_title': item_info.get('title', '') if item_info else '',
                    'item_categories': item_info.get('categories', []) if item_info else [],
                    'item_description': item_info.get('description', '') if item_info else ''
                }

            history.append(purchase_record)


        def sort_key(record):
            timestamp = record.get('unixReviewTime', 0)
            if timestamp is None:
                return 0
            return timestamp

        history.sort(key=sort_key, reverse=True)

        self.logger.info(f"users {user_id} (ID: {original_user_id}) found {len(history)} ")
        return history
    
    def analyze_user_preferences(self, purchase_history: List[Dict]) -> Dict:
        """Analyze user preferences"""
        if not purchase_history:
            return {}
        

        ratings = []
        for h in purchase_history:
            rating = h.get('rating')
            if rating is not None:
                try:

                    numeric_rating = float(rating)
                    ratings.append(numeric_rating)
                except (ValueError, TypeError):

                    continue
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        rating_distribution = Counter(ratings)
        

        all_categories = []
        for h in purchase_history:
            categories = h.get('item_categories', [])
            if isinstance(categories, list):
                for cat in categories:
                    if isinstance(cat, str):
                        all_categories.append(cat)
                    elif isinstance(cat, list):
                        all_categories.extend(cat)
                    elif isinstance(cat, dict):

                        if 'name' in cat:
                            all_categories.append(str(cat['name']))
                        elif 'category' in cat:
                            all_categories.append(str(cat['category']))
                        else:

                            all_categories.append(str(cat))
                    else:

                        all_categories.append(str(cat))
        
        category_counts = Counter(all_categories)
        top_categories = category_counts.most_common(10)
        

        brands = []
        for h in purchase_history:
            brand_info = h.get('brand', '')
            if brand_info:
                if isinstance(brand_info, dict):

                    if 'brand' in brand_info:
                        brands.append(str(brand_info['brand']))
                    elif 'manufacturer' in brand_info:
                        brands.append(str(brand_info['manufacturer']))
                    else:
                        brands.append(str(brand_info))
                else:
                    brands.append(str(brand_info))
        
        brand_counts = Counter(brands)
        top_brands = brand_counts.most_common(5)
        

        total_purchases = len(purchase_history)
        
        return {
            'avg_rating': avg_rating,
            'rating_distribution': dict(rating_distribution),
            'total_purchases': total_purchases,
            'top_categories': top_categories,
            'top_brands': top_brands,
            'purchase_history_sample': purchase_history[:20]
        }
    
    def generate_prompt(self, user_preferences: Dict, user_id: str = None) -> str:
        """Generate prompt for LLM"""
        # print(user_preferences)
        system_prompt = """
# Role: User-Behavior Psychologist & E-commerce Recommendation Specialist

## Profile
- language: English
- description: A specialist in consumer psychology, adept at analyzing e-commerce purchase and review data to construct nuanced, first-person psychological and interest profiles. You bridge the gap between purchasing behavior and deep consumer motivations, preferences, and personality traits.
- background: You have extensive experience working with consumer behavior teams at major e-commerce platforms (like Amazon, eBay). You are trained to see beyond explicit transactions and infer the "why" behind user purchasing decisions to improve product recommendations and customer satisfaction.
- personality: Analytical, empathetic, insightful, and precise. You communicate complex psychological inferences in a conversational, authentic, and easily digestible manner.
- expertise: Consumer behavior analysis, psychological profiling from purchasing patterns, product preference inference, understanding e-commerce recommendation systems.

## Rules

1. Fundamental Principles:
   - Data-Driven Exclusivity: Base the entire profile ONLY on the provided purchase history and review data. Do not use any external knowledge or make assumptions beyond the data.
   - Holistic Interpretation: Synthesize purchase patterns, ratings, review sentiments, and product categories to inform your analysis.
   - Inference over Extraction: Do not simply list purchased products. Your primary task is to infer the psychological drivers, latent needs, and deeper preferences the purchases serve.
   - First-Person Perspective: The entire output must be written in the first person ("I," "My"), as if the user is describing themselves.

2. Behavioral Guidelines:
   - Objectivity and Nuance: Base inferences strictly on evidence from the purchase history. Use nuanced language (e.g., "I seem to be," "I'm likely," "This suggests a preference for") rather than making absolute claims.
   - Authenticity and Conciseness: Maintain a conversational, genuine tone. The total output must be concise, approximately 200-300 words.
   - Empathy: Frame the profile in an insightful and non-judgmental way, focusing on understanding the user's motivations and lifestyle.

3. Restrictions:
   - Language: Write in English ONLY.
   - No External Information: Do not ask for or incorporate any information outside of the provided purchase history.

## Output Format

Please analyze the purchase history and generate a user profile with the following structure:

### My Shopping & Lifestyle Profile

#### Core Interests & Preferences (Ranked)
1. **[Interest Theme] (High/Medium/Low):** [Description of interest based on purchases and reviews]

#### Shopping Behavior Patterns
- [Describe purchasing habits, price sensitivity, brand loyalty, etc.]

#### Product Preferences & Quality Expectations
- [Infer quality expectations from ratings and reviews]
- [Describe preferred product characteristics]

#### Lifestyle & Personal Values (Inferred)
- [Infer lifestyle characteristics from product categories and review patterns]
- [Describe values that drive purchasing decisions]

#### Motivations & Decision Drivers
[2-3 sentences describing what motivates purchasing decisions and how shopping fits into lifestyle]

#### Personal Summary
[1-2 sentences encapsulating the user's overall shopping personality and consumer profile]
"""


        user_data = f"""
Here is my recent purchase and review history for {self.dataset_name} products:

## Review Statistics:
- Total purchases: {user_preferences.get('total_purchases', 0)}
- Average rating given: {user_preferences.get('avg_rating', 0):.2f}/5.0
- Rating distribution: {user_preferences.get('rating_distribution', {})}

## Top Product Categories:
"""
        
        for category, count in user_preferences.get('top_categories', [])[:8]:
            user_data += f"- {category}: {count} purchases\n"
        
        user_data += "\n## Top Brands:\n"
        for brand, count in user_preferences.get('top_brands', [])[:5]:
            user_data += f"- {brand}: {count} purchases\n"
        
        user_data += "\n## Recent Review History (sorted by timestamp in descending order - most recent first):\n"
        user_data += "Note: Reviews are sorted by timestamp in descending order, with the most recent reviews appearing first.\n"
        
        for i, purchase in enumerate(user_preferences.get('purchase_history_sample', [])[:15], 1):
            user_data += f"\n Item {i}: {purchase.get('item_title', 'Unknown Product')}\n"
            user_data += f"   - My Rating: {purchase.get('rating', 'N/A')}/5.0\n"
            # if purchase.get('title'):
            #     user_data += f"   - Review Title: \"{purchase.get('title')}\"\n"
            if purchase.get('review_text'):
                review_text = purchase.get('review_text', '')[:200] + "..." if len(purchase.get('review_text', '')) > 200 else purchase.get('review_text', '')
                user_data += f"   - My Review: \"{review_text}\"\n"
            if purchase.get('item_categories'):
                categories = purchase.get('item_categories', [])
                if categories:
                    cat_str = ", ".join(categories[:3]) if isinstance(categories[0], str) else str(categories)[:100]
                    user_data += f"   - Item Categories: {cat_str}\n"
            if purchase.get('item_description'):
                user_data += f"   - Item Description: {purchase.get('item_description')}\n"

        full_prompt = system_prompt + "\n\n" + user_data + "\n\nPlease generate my first-person consumer profile following the system instructions above."


        if self.save_prompts and user_id:
            try:

                os.makedirs(self.prompts_dir, exist_ok=True)


                prompt_file = os.path.join(self.prompts_dir, f"user_{user_id}_prompt.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(full_prompt)

                self.logger.info(f"saved tofile: {prompt_file}")
            except Exception as e:
                self.logger.error(f"savefilefailed: {e}")

        return full_prompt
    
    async def get_llm_response(self, prompt: str, temperature: float = 0.3) -> str:
        """Get LLM response"""
        return await self.api_client.get_llm_response(prompt, temperature)
    
    def _prepare_data_sync(self, user_id: str) -> Optional[str]:
        """Synchronously execute data preparation, return generated prompt"""
        try:

            purchase_history = self.get_user_purchase_history(user_id)
            if not purchase_history:
                self.logger.warning(f"User {user_id} has no purchase history data")
                return None

            preferences = self.analyze_user_preferences(purchase_history)

            prompt = self.generate_prompt(preferences, user_id)
            return prompt

        except Exception as e:
            self.logger.error(f"User {user_id} data preparation error: {e}")
            return None

    async def extract_user_profile(self, user_id: str, max_retries: int = 10) -> Optional[str]:
        """Extract profile for a single user"""
        try:

            loop = asyncio.get_event_loop()
            prompt = await loop.run_in_executor(self.executor, self._prepare_data_sync, user_id)

            if prompt is None:
                return None

            profile = await self.api_client.get_llm_response_with_retry(prompt, max_retries=max_retries)
            
            if profile and not profile.startswith("Error:"):

                profile = re.sub(r'<think>.*?</think>\s*', '', profile, flags=re.DOTALL)
                return profile
            else:
                self.logger.error(f"users {user_id} generatefailed: {profile}")
                return None

        except Exception as e:
            self.logger.error(f"users {user_id} error: {e}")
            return None
    
    async def batch_extract_profiles(self,
                                   user_ids: List[str] = None,
                                   max_users: int = 50,
                                   output_dir: str = "user_profiles",
                                   prompts_dir: str = None,
                                   max_workers: int = 10) -> Dict[str, str]:
        """Batch extract user profiles"""
        

        if user_ids is None:
            available_users = list(self.review_data.keys())
            user_ids = available_users[:max_users]
        
        self.logger.info(f"Starting processing {len(user_ids)} users")
        

        os.makedirs(output_dir, exist_ok=True)


        if prompts_dir is not None:
            self.prompts_dir = prompts_dir
        

        semaphore = asyncio.Semaphore(max_workers)
        

        profiles = {}
        successful = 0
        failed = 0
        skipped = 0
        

        start_time = time.time()


        pbar = tqdm(total=len(user_ids), desc="users",
                   unit="user", colour="green")
        
        def save_profile_sync(user_id: str, profile: str) -> None:
            """Synchronously save user profile to file"""
            output_file = os.path.join(output_dir, f"user_{user_id}_profile.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(profile)

        async def process_user_with_progress(user_id: str) -> None:
            nonlocal successful, failed, skipped
            
            async with semaphore:
                try:

                    output_file = os.path.join(output_dir, f"user_{user_id}_profile.txt")
                    if os.path.exists(output_file):
                        skipped += 1
                        pbar.set_postfix({"✅": successful, "❌": failed, "⏭️": skipped, "current": f"user_{user_id}_SKIPPED"})
                        print(f"\n⏭️ User {user_id} file already exists, skipping")
                        pbar.update(1)
                        return

                    profile = await self.extract_user_profile(user_id)
                    
                    if profile:

                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(self.executor, save_profile_sync, user_id, profile)

                        profiles[user_id] = profile
                        successful += 1
                        
                        pbar.set_postfix({
                            "✅": successful, 
                            "❌": failed, 
                            "⏭️": skipped, 
                            "current": f"user_{user_id}"
                        })
                        print(f"\n✅ User {user_id} profile saved to {output_dir}/user_{user_id}_profile.txt")
                    else:
                        failed += 1
                        pbar.set_postfix({"✅": successful, "❌": failed, "⏭️": skipped, "current": f"user_{user_id}_FAILED"})
                        print(f"\n❌ User {user_id} profile generation failed")

                except Exception as e:
                    failed += 1
                    self.logger.error(f"Processing user {user_id} error: {e}")
                    pbar.set_postfix({"✅": successful, "❌": failed, "⏭️": skipped, "current": f"user_{user_id}_ERROR"})
                    print(f"\n❌ User {user_id} processing failed: {e}")

                pbar.update(1)
        
        try:

            tasks = [process_user_with_progress(uid) for uid in user_ids]
            await asyncio.gather(*tasks)
        finally:

            pbar.close()
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Statistics: {successful} successful, {skipped} skipped, {failed} failed / {len(user_ids)}")
        print(f"📈 Success rate: {(successful/len(user_ids)):.1%}, Skip rate: {(skipped/len(user_ids)):.1%}")
        print(f"📁 Results saved to: {output_dir}/")
        
        return profiles


def main():
    """Main function example"""

    parser = argparse.ArgumentParser(description="User profile extractor")
    parser.add_argument('--dataset', type=str, default='Beauty_and_Personal_Care', help='Dataset name')
    parser.add_argument('--api_config', type=str, default='simulation/api_config.json', help='API configuration file path')

    args = parser.parse_args()
    extractor = UserProfileExtractor(dataset_name=args.dataset, api_config_path=args.api_config)

    print(f"📋 {len(extractor.interaction_data)} usersdata")
    print(f"📋 {len(extractor.review_data)} usersdata")
    print(f"🔗 User mapping: {len(extractor.user_mapping)}")
    print(f"🔗 Item mapping: {len(extractor.item_mapping)}")

    if extractor.user_mapping:
        mapping_sample = list(extractor.user_mapping.items())[:3]
        print(f"🔍 User mapping: {mapping_sample}")

    if extractor.item_mapping:
        item_mapping_sample = list(extractor.item_mapping.items())[:3]
        print(f"🔍 Item mapping: {item_mapping_sample}")

    interaction_user_ids = list(extractor.interaction_data.keys())
    print(f"🎯 processing {len(interaction_user_ids)} users: {interaction_user_ids}")
    
    async def run_extraction():
        profiles = await extractor.batch_extract_profiles(
            user_ids=interaction_user_ids,
            max_users=len(interaction_user_ids),
            output_dir=f"simulation/user_profiles_{args.dataset}",
            prompts_dir=f"simulation/user_profile_prompts_{args.dataset}",
            max_workers=100
        )

        print(f"\n🎉 {len(profiles)} users")

        if profiles:
            first_user = list(profiles.keys())[0]
            print(f"\n📝 users {first_user} :")
            print("-" * 50)
            profile_preview = profiles[first_user][:500] + "..." if len(profiles[first_user]) > 500 else profiles[first_user]
            print(profile_preview)
    

    asyncio.run(run_extraction())

if __name__ == "__main__":
    main()