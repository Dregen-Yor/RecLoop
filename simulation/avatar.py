#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avatar implementation for simulating recommendation system users.

Features:
- Inject user persona into system prompt during initialization
- Maintain session memory and record historical interactions
- Evaluate each item and return interaction decisions in JSON format
"""

from __future__ import annotations

import os
import re
import json
import time
import datetime
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional
from itertools import cycle

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import APIClient

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMAvatar:
    """API client-based user avatar"""


    _url_cycle = None
    _url_stats = {}


    _shared_api_client = None
    _api_client_config_path = "simulation/api_config.json"

    @classmethod
    def _ensure_api_client_initialized(cls):
        """Ensure shared API client is initialized (execute only once)"""
        if cls._shared_api_client is None:
            try:
                logging.info("🔄 Initializing API client...")
                start_time = time.time()
                cls._shared_api_client = APIClient(cls._api_client_config_path)
                init_time = time.time() - start_time
                logging.info(f"✅ API client initialized successfully (time: {init_time:.2f}s)")
            except Exception as e:
                cls._shared_api_client = None
                logging.warning(f"⚠️ API client initialization failed: {e}")

    def __init__(
        self,
        user_id: str,
        persona_text: str,
        memory_size: int = 5,
        temperature: float = 0.3,
        model_name: str = "gpt-5-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        memory_storage_dir: str = "./user_memory",
        timeout: int = 90,
        max_retries: int = 10,
        reflection_interval: int = 5
    ):
        """
        Initialize avatar

        Args:
            user_id: User ID
            persona_text: User persona text
            memory_size: Memory window size
            temperature: Model temperature parameter
            model_name: Model name to use
            base_url: API base URL (optional, for API compatibility)
            api_key: API key (optional, default read from environment variables)
            memory_storage_dir: Memory file storage directory
            timeout: API request timeout (seconds)
            max_retries: Maximum retry attempts
            reflection_interval: Reflection interval (reflect every N interactions, 0 to disable)
        """
        self.user_id = user_id
        self.persona_text = persona_text
        self.dynamic_persona = persona_text
        self.memory_size = memory_size
        self.memory_storage_dir = memory_storage_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.reflection_interval = reflection_interval
        self.interaction_count = 0


        self.full_conversation_history: List[Dict[str, Any]] = []


        self.api_response_times = []
        self.current_round_response_times = []
        

        self.retry_count = 0
        self.failed_requests = 0


        os.makedirs(memory_storage_dir, exist_ok=True)


        numbert_user_id = user_id.split('_')[-1]
        self.memory_file_path = os.path.join(memory_storage_dir, f"user_{numbert_user_id}_memory.json")


        # api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


        # base_urls = [os.getenv("BASE_URL3")]
        # available_urls = [url for url in base_urls if url]

        # if LLMAvatar._url_cycle is None and available_urls:
        #     LLMAvatar._url_cycle = cycle(available_urls)

        #     for url in available_urls:
        #         LLMAvatar._url_stats[url] = {"success": 0, "failure": 0, "timeouts": 0}



        # if base_url is None and LLMAvatar._url_cycle:
        #     base_url = next(LLMAvatar._url_cycle)


        self.current_base_url = base_url
        # self.available_urls = available_urls


        self._ensure_api_client_initialized()
        self.api_client = self._shared_api_client


        self.memory = ConversationBufferWindowMemory(
            k=memory_size,
            return_messages=True,
            memory_key="chat_history"
        )
        

        self._load_memory_from_file()


        self.system_prompt = self._build_system_prompt()
        

        if not isinstance(self.system_prompt, str):
            logging.error(f"User {self.user_id} system prompt error! Type: {type(self.system_prompt)}")
            self.system_prompt = str(self.system_prompt) if not isinstance(self.system_prompt, list) else "\n".join(str(x) for x in self.system_prompt)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with dynamic user persona"""

        if not isinstance(self.dynamic_persona, str):
            logging.error(f"User {self.user_id}: dynamic persona error! Type: {type(self.dynamic_persona)}, Content: {self.dynamic_persona[:200] if hasattr(self.dynamic_persona, '__getitem__') else self.dynamic_persona}")

            if isinstance(self.dynamic_persona, list):
                self.dynamic_persona = "\n".join(str(item) for item in self.dynamic_persona)
                logging.warning(f"User {self.user_id}: converted dynamic persona from list to string")
            else:
                self.dynamic_persona = str(self.dynamic_persona)
                logging.warning(f"User {self.user_id}: converted dynamic persona to string")
        
        prompt_parts = [
            "You excel at role-playing. Picture yourself as a user exploring a recommender system.",
            "",
            "This is your full user profile (act strictly accordingly):",
            "--- START OF PROFILE ---",
            self.dynamic_persona,
            "--- END OF PROFILE ---"
        ]
        return "\n".join(prompt_parts)
    
    
    def _invoke_with_retry(self, input_data: Dict[str, Any]) -> str:
        """LLM invocation with retry mechanism using exponential backoff strategy"""
        async def async_invoke():
            last_error = None


            prompt = self._build_prompt_from_input(input_data)



            # logging.info(f"{prompt}\n{'='*60}\n")

            for attempt in range(self.max_retries):
                try:

                    response = await self.api_client.get_llm_response(
                        prompt=prompt,
                        temperature=0,
                        timeout=self.timeout
                    )


                    if response and not response.startswith("Error:"):

                        if self.current_base_url in LLMAvatar._url_stats:
                            LLMAvatar._url_stats[self.current_base_url]["success"] += 1
                        return response
                    else:

                        raise Exception(f"Invalid response: {response}")

                except Exception as e:
                    last_error = e
                    error_msg = str(e)


                    if self.current_base_url in LLMAvatar._url_stats:
                        LLMAvatar._url_stats[self.current_base_url]["failure"] += 1
                        if "timed out" in error_msg.lower():
                            LLMAvatar._url_stats[self.current_base_url]["timeouts"] += 1


                    if attempt < self.max_retries - 1:

                        wait_time = min(2 ** attempt, 8)
                        self.retry_count += 1

                        logging.warning(
                            f"User {self.user_id} API call failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                            f"Waiting {wait_time} seconds before retrying..."
                        )
                        await asyncio.sleep(wait_time)
                    else:

                        logging.error(
                            f"User {self.user_id} API call failed after {self.max_retries} attempts: {error_msg}"
                        )


            raise last_error


        try:
            import asyncio
            return asyncio.run(async_invoke())
        except RuntimeError:

            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_invoke())

    def _build_prompt_from_input(self, input_data: Dict[str, Any]) -> str:
        """Build prompt from input data (using recent selections instead of entire memory)"""
        try:
            item_description = input_data.get("item_description", "")

            recent_selections = self.get_recent_selections(n=5)

            if not isinstance(self.system_prompt, str):
                logging.error(f"User {self.user_id}: system prompt error! Type: {type(self.system_prompt)}")
                logging.error(f"User {self.user_id}: system prompt content: {str(self.system_prompt)[:500]}")
                logging.error(f"User {self.user_id}: dynamic persona type: {type(self.dynamic_persona)}")

                self.system_prompt = self._build_system_prompt()
                logging.warning(f"User {self.user_id}: rebuilt system prompt")

            prompt_parts = [self.system_prompt]


            if recent_selections:
                prompt_parts.append(
                    f"\n## Your Recent Selections:\n{recent_selections}"
                )

            valid_item_ids = []
            if 'items_list' in input_data:
                for item in input_data['items_list']:
                    if 'item_id' in item:
                        valid_item_ids.append(str(item['item_id']))


            valid_ids_str = ", ".join(valid_item_ids) if valid_item_ids else "None"

            prompt_parts.extend([
            "\nYou will be presented with a list of items by recommender.",
            "Your task is to SELECT EXACTLY ONE item that you are most interested in based on your profile and preferences.",
            "If NONE of the items interest you, you can choose not to select any item.",
            "",
            f"\n## Recommended Items List ##\n{item_description}",
            "",
            f"## Valid Item IDs (YOU CAN ONLY CHOOSE FROM THESE): [{valid_ids_str}]",
            "",
            "CRITICAL RULES:",
            f"- selected_item_id MUST be one of: {valid_ids_str}, or null if no selection.",
            "- DO NOT use any other numbers (from titles, descriptions, etc.) as Item ID.",
            "- Only select an item that truly aligns with your taste and preferences.",
            "- If nothing interests you, set selected_item_id to null.",
            "",
            "Make decisions that are consistent with your persona and previous interactions.",
            "\nReturn your decision in JSON format:",
            "- reason: Brief explanation of your choice (string)",
            "- selected_item_id: One of the valid IDs above (integer), or null if no selection",
            "",
            ])

            return "\n".join(prompt_parts)
        except Exception as e:
            logging.error(f"buildfailed: {e}")

            try:
                if 'prompt_parts' in locals():
                    logging.error(f"User {self.user_id}: prompt_parts : {len(prompt_parts) if isinstance(prompt_parts, list) else 'N/A'}")
                    if isinstance(prompt_parts, list) and len(prompt_parts) > 0:
                        for i, part in enumerate(prompt_parts[:3]):
                            logging.error(f"User {self.user_id}: prompt_parts[{i}] : {type(part)}, list: {isinstance(part, list)}")
                            if isinstance(part, list):
                                logging.error(f"User {self.user_id}: prompt_parts[{i}] : {part[:3] if len(part) > 3 else part}")
                else:
                    logging.error(f"User {self.user_id}: prompt_parts not found (error)")
                    logging.error(f"User {self.user_id}: self.system_prompt type: {type(self.system_prompt)}")
                    logging.error(f"User {self.user_id}: self.dynamic_persona type: {type(self.dynamic_persona)}")
            except Exception as debug_error:
                logging.error(f"User {self.user_id}: DEBUG failed: {debug_error}")
            return str(input_data)
    
    def _load_memory_from_file(self):
        """Load user memory, dynamic persona, and full conversation history from file"""
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)

                self.full_conversation_history = memory_data.get('full_conversation_history', [])

                memory_messages = memory_data.get('memory_messages', [])
                recent_messages = memory_messages[-(self.memory_size * 2):] if len(memory_messages) > self.memory_size * 2 else memory_messages
                for msg in recent_messages:
                    if msg['type'] == 'recommender':
                        self.memory.chat_memory.add_user_message(msg['content'])
                    elif msg['type'] == 'user':
                        self.memory.chat_memory.add_ai_message(msg['content'])

                saved_dynamic_persona = memory_data.get('dynamic_persona', None)
                if saved_dynamic_persona:
                    self.dynamic_persona = saved_dynamic_persona
                    self.system_prompt = self._build_system_prompt()

                self.interaction_count = memory_data.get('interaction_count', 0)

        except Exception as e:
            logging.error(f"Loading memory for user {self.user_id} failed: {e}")
    
    def _save_memory_to_file(self):
        """Save user memory, dynamic persona, and full conversation history to file"""
        try:
            memory_messages = []
            for msg in self.memory.chat_memory.messages:
                if hasattr(msg, 'content'):
                    msg_type = 'recommender' if isinstance(msg, HumanMessage) else 'user'
                    memory_messages.append({
                        'type': msg_type,
                        'content': msg.content
                    })

            memory_data = {
                'user_id': self.user_id,
                'memory_messages': memory_messages,
                'full_conversation_history': self.full_conversation_history,
                'dynamic_persona': self.dynamic_persona,
                'interaction_count': self.interaction_count,
                'last_updated': datetime.datetime.now().isoformat()
            }

            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logging.error(f"Saving memory for user {self.user_id} failed: {e}")
    
    def decide_items(self, items_list: List[Dict[str,Any]], cycle: int = None) -> Dict[str, Any]:
        """
        Make interaction decision on recommended items list (select the most interesting one from the list)

        Args:
            items_list: List of recommended items
            cycle: Current simulation round (used for recording round)

        Returns:
            Dictionary containing decision information
        """
        try:
            memory_context = self._get_memory_context()
            logging.info(f"Memory context length: {len(memory_context)}")

            items_descriptions = []
            for i, item_info in enumerate(items_list):
                item_desc = f"[Item {i+1}]"
                item_desc += f"\nItem ID: {item_info.get('item_id', '')}"
                item_desc += f"\nTitle: {item_info.get('title', '')}"
                item_desc += f"\nCategories: {item_info.get('categories', [])}"
                item_desc += f"\nDescription: {item_info.get('description', '')}"
                # item_desc += f"\nDetails: {item_info.get('details', '')}"
                # item_desc += f"\nFeatures: {item_info.get('features', '')}"
                items_descriptions.append(item_desc)

            item_description = "\n\n".join(items_descriptions)

            input_data = {
                "item_description": item_description,
                "memory_context": memory_context,
                "items_list": items_list,
            }

            logging.info(f"User {self.user_id} evaluating {len(items_list)} items")

            start_time = time.time()
            response = self._invoke_with_retry(input_data)
            end_time = time.time()
            response_time = end_time - start_time

            self.api_response_times.append(response_time)
            self.current_round_response_times.append(response_time)

            logging.info(f"User {self.user_id} decision completed")

            try:
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    decision = json.loads(json_match.group())
                else:
                    decision = json.loads(response)
            except json.JSONDecodeError:
                decision = self._parse_response_fallback(response, item_description)

            selected_id = decision.get("selected_item_id")
            decision.setdefault("reason", "No reason provided")

            valid_item_ids = set(item.get('item_id') for item in items_list if 'item_id' in item)
            is_valid_selection = True

            if selected_id is not None:
                try:
                    selected_id_int = int(selected_id)
                    if selected_id_int not in valid_item_ids:
                        logging.warning(
                            f"User {self.user_id} selected invalid Item ID: {selected_id}, "
                            f"valid ID list: {list(valid_item_ids)[:5]}... Treating as no selection"
                        )
                        is_valid_selection = False
                        decision["selected_item_id"] = None
                        decision["reason"] = f"[Invalid ID {selected_id} corrected] " + decision.get("reason", "")
                        selected_id = None
                    else:
                        selected_id = selected_id_int
                except (ValueError, TypeError):
                    logging.warning(
                        f"User {self.user_id} selected Item ID with invalid format: {selected_id}, treating as no selection"
                    )
                    is_valid_selection = False
                    decision["selected_item_id"] = None
                    decision["reason"] = f"[Invalid ID format: {selected_id}] " + decision.get("reason", "")
                    selected_id = None

            if selected_id is not None:
                decision["interact"] = True
                decision["item_id"] = int(selected_id)
            else:
                decision["interact"] = False
                decision["item_id"] = -1

            self.interaction_count += 1
            self._add_to_memory(decision, item_description, items_list, cycle=cycle)

            if self.reflection_interval > 0 and self.interaction_count % self.reflection_interval == 0:
                logging.info(f"User {self.user_id} reached interaction count {self.interaction_count}, starting reflection...")
                self.reflect_on_memory()

            return decision
            
        except Exception as e:
            self.failed_requests += 1
            error_msg = str(e)
            
            if "timed out" in error_msg.lower():
                if self.current_base_url in LLMAvatar._url_stats:
                    LLMAvatar._url_stats[self.current_base_url]["timeouts"] += 1
            
            logging.error(
                f"User {self.user_id} decision failed: {error_msg} "
                f"(failure count: {self.failed_requests}, retry count: {self.retry_count})"
            )
            return {
                "selected_item_id": None,
                "item_id": -1,
                "interact": False,
                "reason": f"Decision process failed: {error_msg}"
            }
    
    def _get_memory_context(self) -> str:
        """Get memory context - use recent detailed interaction memories"""

        return self.get_recent_memory_context()
    
    def _parse_response_fallback(self, response: str, item_description: str) -> Dict[str, Any]:
        """Fallback parsing method when JSON parsing fails - new selection format"""

        response_lower = response.lower()
        id_match = re.search(r"(?:select|choose|item|id).*?(\d+)", response_lower)
        selected_item_id = int(id_match.group(1)) if id_match else None

        has_selection = any(word in response_lower for word in ['select', 'choose', 'yes', 'true', 'interested', 'like'])

        if not has_selection or selected_item_id is None:
            decision = {
                "selected_item_id": None,
                "reason": response.strip()[:200] if response else "Not interested in any items"
            }
        else:
            decision = {
                "selected_item_id": selected_item_id,
                "reason": response.strip()[:200] if response else "Selected this item"
            }

        return decision

    def _add_to_memory(self, decision: Dict[str, Any], item_description: str, items_list: List[Dict[str, Any]] = None, cycle: int = None):
        """
        Add interaction record to memory

        Role descriptions:
        - Recommender (recommendation system): Recommends item list
        - User (user played by agent): Makes selection decision

        Args:
            decision: Decision result
            item_description: Item description
            items_list: Recommended item list
            cycle: Current simulation round (if provided, used as round; otherwise use interaction_count)
        """
        selected_id = decision.get('selected_item_id')
        reason = decision.get('reason', 'No reason provided')


        selected_categories = []
        selected_title = ""
        if selected_id is not None and items_list:
            for item in items_list:
                if item.get('item_id') == selected_id:
                    selected_categories = item.get('categories', [])
                    selected_title = item.get('title', '')
                    break


        recommender_message = f"[Recommender] Here are items recommended for you:\n{item_description}"


        if selected_id is not None:
            user_response = f"[User] I choose Item ID: {selected_id}. Reason: {reason}"
        else:
            user_response = f"[User] I'm not interested in any of these items. Reason: {reason}"


        self.memory.chat_memory.add_user_message(recommender_message)
        self.memory.chat_memory.add_ai_message(user_response)


        round_number = cycle if cycle is not None else self.interaction_count


        self.full_conversation_history = [
            r for r in self.full_conversation_history if r.get('round') != round_number
        ]


        self.full_conversation_history.append({
            'round': round_number,
            'recommender': recommender_message,
            'user': user_response,
            'selected_item_id': selected_id,
            'selected_title': selected_title,
            'selected_categories': selected_categories,
            'reason': reason,
            'item_description': item_description,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def get_recent_selections(self, n: int = 5) -> str:
        """
        Get recent n rounds of interaction records (including both selections and non-selections)

        Args:
            n: Get recent n rounds of interactions

        Returns:
            Formatted recent interaction string
        """

        recent = self.full_conversation_history[-n:] if len(self.full_conversation_history) >= n else self.full_conversation_history

        if not recent:
            return ""

        parts = []
        for record in reversed(recent):
            round_num = record.get('round', 1)
            selected_id = record.get('selected_item_id')

            if selected_id is not None:

                parts.append(f"[Round {round_num}]")
                # parts.append(f"Item ID: {selected_id}")
                if record.get('selected_title'):
                    parts.append(f"Title: {record['selected_title']}")
                if record.get('selected_categories'):
                    categories = record['selected_categories']
                    if isinstance(categories, list):

                        if len(categories) > 0 and isinstance(categories[0], list):

                            flat_categories = categories[0]
                            parts.append(f"Categories: {', '.join(str(c) for c in flat_categories)}")
                        else:

                            parts.append(f"Categories: {', '.join(str(c) for c in categories)}")
                    else:
                        parts.append(f"Categories: {categories}")
                parts.append(f"Reason: {record['reason']}")
            else:

                parts.append(f"[Round {round_num}]")
                parts.append("No item selected")
                parts.append(f"Reason: {record['reason']}")

            parts.append("")

        return "\n".join(parts)

    def save_memory(self):
        """Public method: Save user memory to file"""
        self._save_memory_to_file()



    def _collect_conversation_history(self) -> str:
        """
        Collect conversation history between recommendation system and user

        Returns:
            Formatted conversation history string
        """

        if not self.full_conversation_history:
            return ""

        conversation_parts = []

        for record in self.full_conversation_history:
            round_num = record.get('round', 1)
            conversation_parts.append(f"=== Round {round_num} ===")
            conversation_parts.append(record.get('recommender', ''))
            conversation_parts.append(record.get('user', ''))
            conversation_parts.append("")

        return "\n".join(conversation_parts)

    def _build_reflection_prompt(self, conversation_history: str) -> str:
        """
        Build reflection prompt

        Args:
            conversation_history: Conversation history

        Returns:
            Reflection prompt
        """
        prompt = f"""You are analyzing the interaction history between a recommender system and a user.

## User Profile
{self.persona_text}

## Conversation History
{conversation_history}

## Task
Based on the conversation history above, please provide a concise reflection summary that includes:

1. **User Preference Patterns**: What types of items does the user consistently prefer or reject?
2. **Decision Factors**: What factors seem to influence the user's decisions (e.g., categories, features, descriptions)?
3. **Behavioral Trends**: Are there any notable trends or changes in the user's behavior over time?
4. **Recommendations for Future**: What insights can help better serve this user in future recommendations?

Please provide a structured summary in the following JSON format:
{{
    "preference_patterns": "Brief description of user's preference patterns",
    "key_decision_factors": ["factor1", "factor2", ...],
    "behavioral_trends": "Description of any behavioral trends",
    "insights": "Key insights for future recommendations",
    "summary": "One paragraph overall summary"
}}
"""
        return prompt

    async def _async_reflect(self, conversation_history: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronously perform reflection (internal method)

        Args:
            conversation_history: Conversation history

        Returns:
            Reflection result dictionary, returns None on failure
        """
        try:

            reflection_prompt = self._build_reflection_prompt(conversation_history)

            logging.info(f"users {self.user_id} Start, history: {len(conversation_history)}")


            response = await self.api_client.get_llm_response(
                prompt=reflection_prompt,
                temperature=0,
                timeout=self.timeout
            )

            if not response or response.startswith("Error:"):
                logging.warning(f"users {self.user_id} APIfailed: {response}")
                return None


            try:
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    reflection_result = json.loads(json_match.group())
                else:
                    reflection_result = {"summary": response.strip()[:500]}
            except json.JSONDecodeError:
                reflection_result = {"summary": response.strip()[:500]}

                logging.info(f"users {self.user_id} completed")
            return reflection_result

        except Exception as e:
            logging.error(f"users {self.user_id} failed: {e}")
            return None

    def reflect_on_memory(self) -> Optional[Dict[str, Any]]:
        """
        Reflect on memory and generate summary

        Process:
        1. Collect conversation history between recommendation system and user
        2. Call API for summarization
        3. Store summary in memory

        Returns:
            Reflection result dictionary, returns None on failure
        """

        conversation_history = self._collect_conversation_history()

        if not conversation_history:
            logging.info(f"users {self.user_id} history, ")
            return None


        import asyncio
        try:
            reflection_result = asyncio.run(self._async_reflect(conversation_history))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            reflection_result = loop.run_until_complete(self._async_reflect(conversation_history))

        if reflection_result is None:
            return None


        self._store_reflection(reflection_result)

        return reflection_result

    def _store_reflection(self, reflection: Dict[str, Any]):
        """
        Store reflection result to memory and update dynamic persona

        Args:
            reflection: Reflection result dictionary
        """

        summary = reflection.get('summary', '')
        preferences = reflection.get('preference_patterns', '')
        factors = reflection.get('key_decision_factors', [])
        trends = reflection.get('behavioral_trends', '')
        insights = reflection.get('insights', '')

        reflection_content = f"""=== Memory Reflection ===
Summary: {summary}
Preference Patterns: {preferences}
Key Decision Factors: {', '.join(factors) if isinstance(factors, list) else factors}
Behavioral Trends: {trends}
Insights: {insights}
=== End Reflection ==="""


        self.memory.chat_memory.add_user_message("[SYSTEM] Triggering memory reflection")
        self.memory.chat_memory.add_ai_message(reflection_content)


        self._update_dynamic_persona(reflection)

        logging.info(f"users {self.user_id} resultmemory, ")

    def _update_dynamic_persona(self, reflection: Dict[str, Any]):
        """
        Merge reflection result into dynamic persona

        Args:
            reflection: Reflection result dictionary
        """
        summary = reflection.get('summary', '')
        preferences = reflection.get('preference_patterns', '')
        factors = reflection.get('key_decision_factors', [])
        trends = reflection.get('behavioral_trends', '')
        insights = reflection.get('insights', '')


        reflection_summary = f"""

=== Learned Preferences (Updated from interaction history) ===
{f"Preference Patterns: {preferences}" if preferences else ""}
{f"Key Decision Factors: {', '.join(factors) if isinstance(factors, list) else factors}" if factors else ""}
{f"Behavioral Trends: {trends}" if trends else ""}
{f"Insights: {insights}" if insights else ""}
{f"Summary: {summary}" if summary else ""}
=== End Learned Preferences ==="""


        self.dynamic_persona = self.persona_text + reflection_summary


        self.system_prompt = self._build_system_prompt()

        logging.info(f"users {self.user_id} , : {len(self.dynamic_persona)}")

    def get_latest_reflection(self) -> Optional[str]:
        """
        Get the latest reflection content

        Returns:
            Latest reflection content, returns None if not found
        """
        messages = self.memory.chat_memory.messages
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "=== Memory Reflection ===" in msg.content:
                return msg.content
        return None



    def start_new_round(self):
        """Start new recommendation round - reset current round response time statistics"""
        self.current_round_response_times = []

    def get_current_round_response_stats(self):
        """Get current round response time statistics"""
        if not self.current_round_response_times:
            return {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0
            }

        times = self.current_round_response_times
        return {
            "count": len(times),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }

    @classmethod
    def get_url_stats(cls):
        """Get load balancing statistics for all URLs (class method)"""
        if not cls._url_stats:
            return {}
        
        stats_summary = {}
        for url, stats in cls._url_stats.items():
            total = stats["success"] + stats["failure"]
            success_rate = stats["success"] / total if total > 0 else 0
            stats_summary[url] = {
                "success": stats["success"],
                "failure": stats["failure"],
                "timeouts": stats["timeouts"],
                "total": total,
                "success_rate": f"{success_rate:.2%}"
            }
        return stats_summary
    
    @classmethod
    def print_url_stats(cls):
        """Print URL load balancing statistics"""
        stats = cls.get_url_stats()
        if not stats:
            logging.info("URLstats")
            return
        
        logging.info("=" * 60)
        logging.info("URLstats")
        logging.info("=" * 60)
        for url, data in stats.items():
            logging.info(f"\nURL: {url}")
            logging.info(f" : {data['success']}")
            logging.info(f"  failed: {data['failure']}")
            logging.info(f" : {data['timeouts']}")
            logging.info(f" : {data['total']}")
            logging.info(f" : {data['success_rate']}")
        logging.info("=" * 60)
    
    def get_total_response_stats(self):
        """Get total response time statistics"""
        if not self.api_response_times:
            return {
                "total_count": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0
            }

        times = self.api_response_times
        return {
            "total_count": len(times),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }

    def get_memory_summary(self) -> List[str]:
        """Get current memory summary"""
        messages = self.memory.chat_memory.messages
        summaries = []
        for msg in messages:
            if hasattr(msg, 'content') and 'Item' in msg.content:
                summaries.append(msg.content)
        return summaries[-self.memory_size:]
    

    
    def get_recent_memory_context(self) -> str:
        """Get context of the most recent memory_size interactions - includes complete user feedback and item metadata"""
        try:
            messages = self.memory.chat_memory.messages

            recent_messages = messages[-(self.memory_size * 2):] if len(messages) > self.memory_size * 2 else messages
            
            context_parts = []
            i = 0
            while i < len(recent_messages) - 1:
                recomender_msg = recent_messages[i] if i < len(recent_messages) else None
                agent_msg = recent_messages[i + 1] if i + 1 < len(recent_messages) else None

                if recomender_msg and agent_msg and hasattr(recomender_msg, 'content') and hasattr(agent_msg, 'content'):



                    interaction_context = f"Previous interaction (sorted by timestamp in descending order - most recent first):[{recomender_msg.content}]\\n"
                    context_parts.append(interaction_context)
                
                i +=1
            
            return "\n".join(context_parts[-self.memory_size:])
        except Exception as e:
            logging.error(f"failed: {e}")
            return ""
    
    def reset_memory(self):
        """Reset memory"""
        self.memory.clear()


def create_avatar(
    user_id: str,
    persona_text: str,
    memory_size: int = 5,
    memory_storage_dir: str = "./user_memory",
    **llm_kwargs
):
    """
    Factory function to create an avatar
    
    Args:
        user_id: User ID
        persona_text: User persona text
        memory_size: Memory size
        memory_storage_dir: Memory file storage directory
        **llm_kwargs: Additional parameters passed to LLM
        
    Returns:
        Avatar instance
    """
    return LLMAvatar(user_id, persona_text, memory_size, memory_storage_dir=memory_storage_dir, **llm_kwargs)
