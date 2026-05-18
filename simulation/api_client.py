#!/usr/bin/env python3
"""
API Client Class
Encapsulates all API calling functions, including endpoint testing, model queries, inference performance testing, etc.
"""

import json
import asyncio
import time
import random
import httpx
from typing import List, Dict, Optional, Tuple
import openai
import logging


class APIClient:
    """API client class, encapsulates all API calling functions"""
    
    def __init__(self, config_path: str = "api_config.json"):
        """
        Initialize API client
        
        Args:
            config_path: API configuration file path
        """
        self.config_path = config_path
        self.api_key = None
        self.base_urls = []
        self.clients = []
        self.endpoint_models = {}
        
        self.logger = logging.getLogger(__name__)
        
        self._load_config()
        
    def _load_config(self):
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.api_key = config['api_key']
                self.base_urls = config['base_urls']

            self.endpoint_models = config.get('endpoint_models', {})
            # if self.endpoint_models:
            # else:

            self.clients = []
            for base_url in self.base_urls:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=base_url
                )
                self.clients.append(client)

        except Exception as e:
            self.logger.error(f"❌ Config file loading failed: {e}")
            raise

    def get_random_client(self) -> openai.OpenAI:
        """Get random client"""
        return random.choice(self.clients)

    def get_client_by_url(self, base_url: str) -> Optional[openai.OpenAI]:
        """Get client by URL"""
        try:
            return openai.OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
        except Exception as e:
            self.logger.error(f"Client creation failed for {base_url}: {e}")
            return None

    async def list_models(self, base_url: str, timeout: int = 10) -> Dict:
        """List models supported by endpoint"""
        result = {
            'base_url': base_url,
            'status': 'unknown',
            'models': [],
            'error': None
        }

        try:

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )

                if response.status_code == 200:
                    data = response.json()
                    models = [model['id'] for model in data.get('data', [])]
                    result['status'] = 'success'
                    result['models'] = models
                    self.logger.info(f"✅ {base_url} - found {len(models)} models")
                else:
                    result['status'] = 'error'
                    result['error'] = f"HTTP {response.status_code}: {response.text}"
                    self.logger.error(f"❌ {base_url} - model listing failed: {response.status_code}")

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"❌ {base_url} - listing models error: {e}")

        return result

    async def test_endpoint_connection(self, base_url: str, model: str = "Qwen3-8B", timeout: int = 10) -> Dict:
        """Test single endpoint connection availability"""
        result = {
            'base_url': base_url,
            'status': 'unknown',
            'response_time': None,
            'error': None
        }

        try:
            client = self.get_client_by_url(base_url)
            if not client:
                result['status'] = 'connection_error'
                result['error'] = "Failed to create client"
                return result

            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello, test message"}],
                    max_tokens=10,
                    timeout=timeout
                )
            )
            response_time = time.time() - start_time

            result['status'] = 'available'
            result['response_time'] = round(response_time, 2)
            self.logger.info(f"✅ {base_url} - connected (time: {result['response_time']}s)")

        except openai.APIError as e:
            result['status'] = 'api_error'
            result['error'] = str(e)
            self.logger.error(f"❌ {base_url} - API error: {e}")
        except openai.APITimeoutError as e:
            result['status'] = 'timeout'
            result['error'] = str(e)
            self.logger.warning(f"⏱️ {base_url} - timeout: {e}")
        except openai.APIConnectionError as e:
            result['status'] = 'connection_error'
            result['error'] = str(e)
            self.logger.error(f"🔌 {base_url} - connection error: {e}")
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"❌ {base_url} - test error: {e}")

        return result

    async def test_endpoint_inference(self, base_url: str, model: str = "Qwen3-8B", timeout: int = 30) -> Dict:
        """Test single endpoint model inference performance"""
        result = {
            'base_url': base_url,
            'status': 'unknown',
            'inference_time': None,
            'tokens_per_second': None,
            'total_tokens': None,
            'prompt_tokens': None,
            'completion_tokens': None,
            'error': None
        }

        try:
            client = self.get_client_by_url(base_url)
            if not client:
                result['status'] = 'connection_error'
                result['error'] = "Failed to create client"
                return result


            test_prompt = """Please explain in detail the development history of artificial intelligence, including the following important stages:
1. Early development (1950s-1960s)
2. First AI winter (1970s)
3. Expert systems era (1980s)
4. Rise of machine learning (1990s-2000s)
5. Deep learning revolution (2010s to present)

Please provide specific technical breakthroughs, representative figures, and important events for each stage."""


            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=500,
                    temperature=0.7,
                    timeout=timeout
                )
            )
            end_time = time.time()
            
            inference_time = end_time - start_time
            

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            

            tokens_per_second = completion_tokens / inference_time if inference_time > 0 else 0

            result['status'] = 'available'
            result['inference_time'] = round(inference_time, 2)
            result['tokens_per_second'] = round(tokens_per_second, 2)
            result['total_tokens'] = total_tokens
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            
            self.logger.info(f"✅ {base_url} - inference: {result['inference_time']}s, generation: {result['tokens_per_second']} tokens/s, total tokens: {total_tokens}")

        except openai.APIError as e:
            result['status'] = 'api_error'
            result['error'] = str(e)
            self.logger.error(f"❌ {base_url} - API error: {e}")
        except openai.APITimeoutError as e:
            result['status'] = 'timeout'
            result['error'] = str(e)
            self.logger.warning(f"⏱️ {base_url} - timeout: {e}")
        except openai.APIConnectionError as e:
            result['status'] = 'connection_error'
            result['error'] = str(e)
            self.logger.error(f"🔌 {base_url} - connection error: {e}")
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"❌ {base_url} - inference error: {e}")

        return result

    async def list_all_models(self, timeout: int = 10) -> List[Dict]:
        """List models supported by all endpoints"""
        self.logger.info(f"\n🔍 Starting model listing for {len(self.base_urls)} API endpoints...")
        self.logger.info("=" * 60)

        results = []
        tasks = [self.list_models(url, timeout) for url in self.base_urls]
        results = await asyncio.gather(*tasks)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 Model listing results:")

        success_count = sum(1 for r in results if r['status'] == 'success')
        total_count = len(results)

        self.logger.info(f"✅ Success: {success_count}/{total_count}")
        self.logger.info(f"❌ Failed: {total_count - success_count}/{total_count}")

        self.logger.info("\n📋 Detailed results:")
        all_models = set()
        for result in results:
            status_icon = '✅' if result['status'] == 'success' else '❌'
            model_count = len(result['models'])
            self.logger.info(f"{status_icon} {result['base_url']} - {model_count} models")

            if result['models']:
                models_preview = ', '.join(result['models'][:10]) + ("..." if len(result['models']) > 10 else "")
                self.logger.info(f"   Models: {models_preview}")
                all_models.update(result['models'])

            if result['error']:
                error_preview = result['error'][:100] + "..." if len(result['error']) > 100 else result['error']
                self.logger.info(f"   Error: {error_preview}")

                self.logger.info(f"\n🎯 Total unique models: {len(all_models)}")
        if all_models:
            self.logger.info("📝 All unique models:")
            for model in sorted(all_models):
                self.logger.info(f"   - {model}")

        return results

    async def test_all_endpoints_connection(self, timeout: int = 10) -> List[Dict]:
        """Test connection availability of all endpoints"""
        self.logger.info(f"\n🔍 Start {len(self.base_urls)} API...")
        self.logger.info("=" * 60)

        results = []
        tasks = []
        for url in self.base_urls:

            if url in self.endpoint_models:
                endpoint_model = self.endpoint_models[url]['model']
                tasks.append(self.test_endpoint_connection(url, endpoint_model, timeout))
            else:
                self.logger.error(f" {url} filefoundmodelmapping")
        results = await asyncio.gather(*tasks)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 result:")

        available_count = sum(1 for r in results if r['status'] == 'available')
        total_count = len(results)

        self.logger.info(f"✅ : {available_count}/{total_count}")
        self.logger.info(f"❌ : {total_count - available_count}/{total_count}")


        self.logger.info("\n📋 result:")
        for result in results:
            status_icon = {
                'available': '✅',
                'api_error': '❌',
                'timeout': '⏱️',
                'connection_error': '🔌',
                'error': '❌',
                'unknown': '❓'
            }.get(result['status'], '❓')

            time_info = f" ({result['response_time']}s)" if result['response_time'] else ""
            self.logger.info(f"{status_icon} {result['base_url']} - {result['status']}{time_info}")

            if result['error']:
                self.logger.info(f" error: {result['error']}")

        return results

    async def test_all_endpoints_inference(self, timeout: int = 30) -> List[Dict]:
        """Test model inference performance of all endpoints"""
        self.logger.info(f"\n🔍 Start {len(self.base_urls)} APImodelinference...")
        self.logger.info("=" * 60)

        results = []
        tasks = []
        for url in self.base_urls:

            if url in self.endpoint_models:
                endpoint_model = self.endpoint_models[url]['model']
                tasks.append(self.test_endpoint_inference(url, endpoint_model, timeout))
            else:
                self.logger.error(f" {url} filefoundmodelmapping")
        results = await asyncio.gather(*tasks)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 modelinferenceresult:")

        available_count = sum(1 for r in results if r['status'] == 'available')
        total_count = len(results)

        self.logger.info(f"✅ : {available_count}/{total_count}")
        self.logger.info(f"❌ : {total_count - available_count}/{total_count}")


        self.logger.info("\n📋 result:")
        for result in results:
            status_icon = {
                'available': '✅',
                'api_error': '❌',
                'timeout': '⏱️',
                'connection_error': '🔌',
                'error': '❌',
                'unknown': '❓'
            }.get(result['status'], '❓')

            if result['status'] == 'available':
                perf_info = f" (inference: {result['inference_time']}s, speed: {result['tokens_per_second']} tokens/s)"
                self.logger.info(f"{status_icon} {result['base_url']} - {result['status']}{perf_info}")
            else:
                self.logger.info(f"{status_icon} {result['base_url']} - {result['status']}")

            if result['error']:
                self.logger.info(f" error: {result['error']}")


        available_results = [r for r in results if r['status'] == 'available']
        if available_results:
            self.logger.info("\n🏆 inference (generate):")
            sorted_results = sorted(available_results, key=lambda x: x['tokens_per_second'], reverse=True)
            for i, result in enumerate(sorted_results, 1):
                self.logger.info(f"{i}. {result['base_url']} - {result['tokens_per_second']} tokens/s (inference: {result['inference_time']}s)")

        return results

    def get_available_endpoints(self, results: List[Dict]) -> List[str]:
        """Get list of all available endpoints"""
        return [r['base_url'] for r in results if r['status'] == 'available']

    def get_best_endpoint(self, results: List[Dict], metric: str = 'tokens_per_second') -> Optional[str]:
        """Get best endpoint"""
        available_results = [r for r in results if r['status'] == 'available']
        if not available_results:
            return None
        
        if metric == 'tokens_per_second':
            best_result = max(available_results, key=lambda x: x.get('tokens_per_second', 0))
        elif metric == 'inference_time':
            best_result = min(available_results, key=lambda x: x.get('inference_time', float('inf')))
        elif metric == 'response_time':
            best_result = min(available_results, key=lambda x: x.get('response_time', float('inf')))
        else:
            return None
            
        return best_result['base_url']

    async def get_llm_response(self, prompt: str, temperature: float = 0, timeout: int = 120) -> str:
        """Get LLM response"""
        client = self.get_random_client()
        

        client_url = str(client.base_url).rstrip('/')
        

        if client_url and client_url in self.endpoint_models:
            model = self.endpoint_models[client_url]['model']
        else:
            self.logger.error(f" {client_url} filefoundmodelmapping")
            self.logger.error(f": {list(self.endpoint_models.keys())}")
            return f"Error: No model mapping found for endpoint {client_url}"
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM APIfailed: {e}")
            return f"Error: Failed to get LLM response - {e}"

    async def get_llm_response_with_retry(self, prompt: str, temperature: float = 0, 
                                        timeout: int = 120, max_retries: int = 10) -> str:
        """Get LLM response with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = await self.get_llm_response(prompt, temperature, timeout)
                if response and not response.startswith("Error:"):
                    return response
                elif attempt < max_retries - 1:
                    self.logger.warning(f"{attempt + 1}failed, ...")
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"{attempt + 1}APIfailed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return f"Error: Failed to get LLM response after {max_retries} attempts"


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API endpoint testing tool")
    parser.add_argument('--config', type=str, default='api_config.json', help='APIfilepath')
    parser.add_argument('--timeout', type=int, default=30, help='(s)')
    parser.add_argument('--models', type=str, nargs='+', help='model, model(, modelfile)')
    parser.add_argument('--list-models', action='store_true', help='model')
    parser.add_argument('--test-type', type=str, default='inference', 
                       choices=['connection', 'models', 'inference'], 
                       help=': connection(), models(model), inference(inference)')

    args = parser.parse_args()

    try:

        client = APIClient(args.config)

        if args.list_models:

            await client.list_all_models(args.timeout)
        else:

            if args.test_type == 'connection':
                print(f"\n🧪 ")
                results = await client.test_all_endpoints_connection(timeout=args.timeout)
            elif args.test_type == 'models':
                print(f"\n🧪 model")
                results = await client.list_all_models(args.timeout)
            elif args.test_type == 'inference':
                print(f"\n🧪 inference")
                results = await client.test_all_endpoints_inference(timeout=args.timeout)
            

            available_endpoints = client.get_available_endpoints(results)
            if available_endpoints:
                print(f"\n💡 : {available_endpoints}")
                

                if args.test_type == 'inference':
                    best_endpoint = client.get_best_endpoint(results)
                    if best_endpoint:
                        print(f"🏆 : {best_endpoint}")
            else:
                print("\n❌ ")

    except Exception as e:
        print(f"❌ error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    exit_code = asyncio.run(main())
    exit(exit_code)
