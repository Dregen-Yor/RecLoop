#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommenders'))

from persistent_user_simulator import UserManager
from data_manager import InteractionDataManager, IncrementalTrainingDataGenerator
from item_info_retriever import ItemInfoRetriever


class ClosedLoopRecommendationSystem:
    """Closed-loop recommendation system main controller"""

    def __init__(self,
                 dataset_name: str = "Toys_and_Games",
                 backbone: str = "SASRec",
                 num_users: int = None,
                 items_per_cycle: int = 5,
                 max_cycles: int = 10,
                 recommendation_file: str = None,
                 recommend_script: str = None,
                 storage_dir: str = "./simulation_storage",
                 enable_training: bool = True,
                 enable_logging: bool = True,
                 max_concurrent_users: int = 5,
                 start_cycle: int = 1,
                 reflection_interval: int = 5):
        """Initialize closed-loop recommendation system"""
        self.dataset_name = dataset_name
        self.backbone = backbone
        self.num_users = num_users
        self.items_per_cycle = items_per_cycle
        self.max_cycles = max_cycles
        self.enable_training = enable_training
        self.enable_logging = enable_logging
        self.max_concurrent_users = max_concurrent_users
        self.start_cycle = start_cycle
        self.reflection_interval = reflection_interval

        self.storage_dir = Path(storage_dir)


        self.recommendation_file_path = recommendation_file
        self.recommend_script = recommend_script

        os.makedirs(self.storage_dir, exist_ok=True)
        data_dir = str(self.storage_dir) + "/data"
        os.makedirs(data_dir, exist_ok=True)


        dst_file = data_dir + f"/{self.dataset_name}-1.txt"
        if start_cycle == 1:
            src_file = Path(f"./recommenders/data/{self.dataset_name}/{self.dataset_name}.txt")
            shutil.copy2(str(src_file), str(dst_file))
        else:

            expected_data_file = data_dir + f"/{self.dataset_name}-{start_cycle}.txt"
            if not os.path.exists(expected_data_file):
                print(f"⚠ : Round {start_cycle} data file does not exist: {expected_data_file}")
                print(f" data file")

        self.user_manager = None
        self.data_manager = None
        self.incremental_generator = None
        self.item_retriever = None

        self.current_cycle = start_cycle
        self.is_initialized = False
        self.experiment_id = None

        self.performance_metrics = {
            "cycles_completed": 0,
            "total_interactions": 0,
            "start_time": None,
            "end_time": None,
            "cycle_metrics": []
        }

        self._vllm_process = None
        self._setup_logging()
        print(f"✓ recommendation completed - dataset: {dataset_name}")

    def _setup_logging(self):
        """Set up enhanced logging system - record all information to local files"""
        if not self.enable_logging:
            return


        log_dir = self.storage_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)
        

        timestamp = int(time.time())
        self.log_files = {
            'main': log_dir / f"simulation_main_{timestamp}.log",
            'user_interactions': log_dir / f"user_interactions_{timestamp}.log",
            'recommendations': log_dir / f"recommendations_{timestamp}.log",
            'data_updates': log_dir / f"data_updates_{timestamp}.log",
            'training': log_dir / f"training_{timestamp}.log",
            'system_stats': log_dir / f"system_stats_{timestamp}.log",
            'errors': log_dir / f"errors_{timestamp}.log"
        }


        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_files['main'], encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        

        self.loggers = {}
        for log_type, log_file in self.log_files.items():
            if log_type != 'main':
                logger = logging.getLogger(f"simulation.{log_type}")
                logger.setLevel(logging.DEBUG)
                handler = logging.FileHandler(log_file, encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
                logger.propagate = False
                self.loggers[log_type] = logger
        
                self.logger.info("")
                self.logger.info(f"log directory: {log_dir}")
        for log_type, log_file in self.log_files.items():
            self.logger.info(f"  {log_type}: {log_file}")
            

        self._log_system_config()

    def _log_system_config(self):
        """Log system initial configuration information"""
        config_info = {
            "dataset_name": self.dataset_name,
            "num_users": self.num_users,
            "items_per_cycle": self.items_per_cycle,
            "max_cycles": self.max_cycles,
            "enable_training": self.enable_training,
            "storage_dir": str(self.storage_dir),
            "recommendation_file_path": self.recommendation_file_path,
            "recommend_script": self.recommend_script
        }
        
        self.logger.info("=== System Configuration ===")
        for key, value in config_info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 50)

    def _log_detailed_message(self, log_type: str, level: str, message: str, extra_data: dict = None):
        """Log detailed message to specified log file"""
        if not self.enable_logging or log_type not in self.loggers:
            return
            
        logger = self.loggers[log_type]
        log_message = message
        
        if extra_data:
            log_message += f" | Detailed data: {json.dumps(extra_data, ensure_ascii=False, indent=2)}"
        
        if level.upper() == 'DEBUG':
            logger.debug(log_message)
        elif level.upper() == 'INFO':
            logger.info(log_message)
        elif level.upper() == 'WARNING':
            logger.warning(log_message)
        elif level.upper() == 'ERROR':
            logger.error(log_message)
        else:
            logger.info(log_message)

    def initialize_system(self) -> bool:
        """Initialize system components"""
        try:
            print("\n=== System Initialization ===")
            self.logger.info("Initializing system components")

            self.user_manager = UserManager(memory_storage_dir=str(self.storage_dir)+"/user_memory_"+self.dataset_name)
            print("✓ User manager initialized")
            self.logger.info("User manager initialized")

            self.data_manager = InteractionDataManager(
                dataset_name=self.dataset_name,
                storage_dir=str(self.storage_dir)
            )
            print("✓ Data manager initialized")
            self.logger.info("Data manager initialized")

            self.incremental_generator = IncrementalTrainingDataGenerator(
                self.user_manager,
                self.data_manager
            )
            print("✓ Incremental data generator initialized")
            self.logger.info("Incremental data generator initialized")

            try:
                dataset_path = f"./recommenders/data"
                    
                if os.path.exists(dataset_path):
                    self.item_retriever = ItemInfoRetriever(dataset_path=dataset_path, dataset_name=self.dataset_name)
                    print(f"✓ Item info retriever initialized")
                    self.logger.info(f"Item info retriever initialized: {dataset_path}")
                else:
                    print(f": Dataset path not found: {dataset_path}")
                    self.logger.warning(f"Dataset path not found: {dataset_path}")
            except Exception as e:
                print(f"Item info retriever initialization failed: {e}")
                self.logger.error(f"Item info retriever initialization failed: {e}")


            self._initialize_users()
            print("✓ Users initialized")

            self.is_initialized = True
            self.experiment_id = f"exp_{int(time.time())}"

            print(f"✓ System initialization completed, Experiment ID: {self.experiment_id}")
            self.logger.info(f"System initialization completed, Experiment ID: {self.experiment_id}")
            

            init_stats = {
                "experiment_id": self.experiment_id,
                "total_users_created": len(self.user_manager.get_all_users()),
                "target_user_count": self.num_users,
                "initialization_success": True
            }
            self._log_detailed_message('system_stats', 'INFO', "System initialization statistics", init_stats)
            
            return True

        except Exception as e:
            print(f"System initialization failed: {e}")
            self.logger.error(f"System initialization failed: {e}", exc_info=True)
            self._log_detailed_message('errors', 'ERROR', f"System initialization failed: {e}")
            return False

    def _initialize_users(self):
        """Initialize users"""
        print(f"\n--- Initializing {self.num_users} users ---")
        self.logger.info(f"Starting to initialize {self.num_users} users")

        persona_dir = self._find_persona_directory()
        successful_users = []
        failed_users = []

        for i in range(1, self.num_users + 1):
            user_id = f"user_{i}"
            try:
                persona_text = self._get_persona_text(i, persona_dir)

                user = self.user_manager.create_or_load_user(
                    user_id=user_id,
                    persona_text=persona_text,
                    temperature=0.3,
                    memory_size=5
                )

                successful_users.append({
                    "user_id": user_id,
                    "user_index": i,
                    "total_interactions": user.total_interactions,
                    "persona_loaded": True
                })
                
            except Exception as e:
                failed_users.append({
                    "user_id": user_id,
                    "user_index": i,
                    "error": str(e),
                    "persona_loaded": False
                })
                print(f"User {user_id} initialization failed: {e}")
                self.logger.warning(f"User {user_id} initialization failed: {e}")
                self._log_detailed_message('errors', 'WARNING', f"User creation failed", {
                    "user_id": user_id,
                    "user_index": i,
                    "error": str(e)
                })


        init_summary = {
            "target_users": self.num_users,
            "successful_users": len(successful_users),
            "failed_users": len(failed_users),
            "success_rate": len(successful_users) / self.num_users * 100,
            "persona_directory": persona_dir,
            "successful_user_list": successful_users,
            "failed_user_list": failed_users
        }
        
        self.logger.info(f"User initialization completed: {len(successful_users)}/{self.num_users} ({len(successful_users)/self.num_users:.1%})")
        self._log_detailed_message('system_stats', 'INFO', "User initialization detailed statistics", init_summary)

    def _find_persona_directory(self) -> str:
        """Find user persona directory"""
        user_profile_dir = "./simulation/user_profiles_"+self.dataset_name
        if os.path.exists(user_profile_dir):
            return user_profile_dir
        
        return ""

    def _get_persona_text(self, user_index: int, persona_dir: str) -> str:
        """Get user persona text - read corresponding profile file based on user ID"""
        if persona_dir:

            persona_file_path = os.path.join(persona_dir, f"user_{user_index}_profile.txt")
            
            if os.path.exists(persona_file_path):
                try:
                    with open(persona_file_path, 'r', encoding='utf-8') as f:
                        profile_text = f.read().strip()
                        if profile_text:
                            return profile_text
                except Exception as e:
                    print(f"Failed to load user profile file {persona_file_path}: {e}")
            else:
                print(f"User profile file does not exist: {persona_file_path}")


                raise Exception(f"User profile file does not exist: {persona_file_path}")

    def run_closed_loop_simulation(self) -> Dict[str, Any]:
        """Run closed-loop simulation"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized, please call initialize_system() first")

        print(f"\n{'='*60}")
        print(f"Starting recommendation system - Experiment ID: {self.experiment_id}")
        if self.start_cycle > 1:
            print(f"Resuming from Round {self.start_cycle} ")
        print(f"{'='*60}")

        self.performance_metrics["start_time"] = time.time()

        try:
            for cycle in range(self.start_cycle, self.max_cycles + 1):
                self.current_cycle = cycle
                print(f"\n{'='*40}")
                print(f"Starting Round {cycle}/{self.max_cycles} ")
                print(f"{'='*40}")

                cycle_start_time = time.time()


                self._ensure_cycle_data_exists(cycle)
                recommendations_data = self._run_recommendation_cycle(cycle)

                self._start_vllm_server()
                interactions_data = self._run_interaction_cycle(cycle, recommendations_data)


                self._stop_vllm_server()

                data_update_stats = {"cycle": cycle, "status": "deferred"}

                training_stats = None
                if self.enable_training and cycle < self.max_cycles:
                    training_stats = self._run_training_cycle(cycle)

                cycle_time = time.time() - cycle_start_time
                cycle_metrics = {
                    "cycle": cycle,
                    "duration": round(cycle_time, 2),
                    "recommendations_count": len(recommendations_data),
                    "interactions_count": sum(len(interactions.get('evaluations', []))
                                             for interactions in interactions_data),
                    "new_interactions": sum(len(interactions.get('selected_items', []))
                                           for interactions in interactions_data),
                    "data_update_stats": data_update_stats,
                    "training_stats": training_stats
                }

                self.performance_metrics["cycle_metrics"].append(cycle_metrics)
                print(f"✓ Round {cycle} completed, Duration: {cycle_time:.2f}s")
                self._save_cycle_checkpoint(cycle, cycle_metrics)

            self.performance_metrics["end_time"] = time.time()
            self.performance_metrics["cycles_completed"] = self.max_cycles

            print(f"\n{'='*60}")
            print("Simulation completed！")
            print(f"{'='*60}")

            return self._generate_final_report()

        except Exception as e:
            print(f"Simulation error: {e}")
            if self.enable_logging:
                self.logger.error(f"Simulation error: {e}", exc_info=True)
            raise

    def _run_recommendation_cycle(self, cycle: int) -> List[Dict[str, Any]]:
        """Run recommendation cycle - run gen_recommend.sh to generate recommendations, then read results from file"""
        print(f"\n--- Round {cycle} recommendation ---")
        self.logger.info(f"Starting Round {cycle} recommendation")


        if not self._generate_recommendations():
            print(f"Recommendation generation failed, exiting")
            self.logger.error(f"Recommendation generation failed, exiting")
            self._log_detailed_message('recommendations', 'ERROR', f"Round {cycle} recommendation generation failed, program terminated")
            sys.exit(1)

        recommendations_data = []
        

        recommendations_dict = self._load_recommendations_from_file()
        
        if not recommendations_dict:
            print(f": Failed to load recommendation results from file {self.recommendation_file_path}")
            self.logger.warning(f"Failed to load recommendation results from file: {self.recommendation_file_path}")
            self._log_detailed_message('recommendations', 'WARNING', "Failed to load recommendation file", {
                "file_path": self.recommendation_file_path,
                "cycle": cycle
            })
            return recommendations_data


        file_stats = {
            "cycle": cycle,
            "total_users_in_file": len(recommendations_dict),
            "total_recommendations": sum(len(recs) for recs in recommendations_dict.values()),
            "file_path": self.recommendation_file_path
        }
        self._log_detailed_message('recommendations', 'INFO', "Recommendation file statistics", file_stats)

        successful_users = []
        failed_users = []

        for user in self.user_manager.get_all_users():
            try:

                user_numeric_id = int(user.user_id.split('_')[1])


                user_recommendations = recommendations_dict.get(user_numeric_id, [])
                
                if not user_recommendations:
                    print(f"User {user.user_id}: Pre-generated recommendations not found")
                    self.logger.warning(f"User {user.user_id}: Pre-generated recommendations not found")
                    failed_users.append({
                        "user_id": user.user_id,
                        "user_numeric_id": user_numeric_id,
                        "reason": "Pre-generated recommendations not found"
                    })
                    continue
                

                limited_recommendations = user_recommendations[:self.items_per_cycle]
                

                enriched_recommendations = self._enrich_recommendations_with_item_info(limited_recommendations)
                
                if enriched_recommendations:
                    user_rec_data = {
                        "user_id": user.user_id,
                        "recommendations": enriched_recommendations,
                        "cycle": cycle,
                        "timestamp": time.time()
                    }
                    recommendations_data.append(user_rec_data)

                    successful_users.append({
                        "user_id": user.user_id,
                        "user_numeric_id": user_numeric_id,
                        "recommendations_count": len(enriched_recommendations),
                        "original_count": len(user_recommendations)
                    })

                    rec_details = {
                        "user_id": user.user_id,
                        "cycle": cycle,
                        "recommendations_count": len(enriched_recommendations),
                        "recommendations": [{"item_id": r.get('item_id'), "title": r.get('title', '')[:50]} for r in enriched_recommendations[:5]]
                    }
                    self._log_detailed_message('recommendations', 'DEBUG', f"User recommendation details", rec_details)
                else:
                    print(f"User {user.user_id}: Recommendation enrichment failed")
                    self.logger.warning(f"User {user.user_id}: Recommendation enrichment failed")
                    failed_users.append({
                        "user_id": user.user_id,
                        "user_numeric_id": user_numeric_id,
                        "reason": "Recommendation enrichment failed"
                    })

            except Exception as e:
                print(f"User {user.user_id} recommendation processing failed: {e}")
                self.logger.error(f"User {user.user_id} recommendation processing failed: {e}")
                failed_users.append({
                    "user_id": user.user_id,
                    "error": str(e),
                    "reason": "Processing exception"
                })
                self._log_detailed_message('errors', 'ERROR', f"User recommendation processing failed", {
                    "user_id": user.user_id,
                    "cycle": cycle,
                    "error": str(e)
                })
                continue


        recommendation_stats = {
            "cycle": cycle,
            "total_users_attempted": len(self.user_manager.get_all_users()),
            "successful_users": len(successful_users),
            "failed_users": len(failed_users),
            "success_rate": len(successful_users) / len(self.user_manager.get_all_users()) * 100 if self.user_manager.get_all_users() else 0,
            "total_recommendations_generated": len(recommendations_data),
            "successful_user_details": successful_users,
            "failed_user_details": failed_users
        }

        print(f"✓ Recommendation cycle completed, processing {len(recommendations_data)} recommendations")
        self.logger.info(f"Round {cycle} recommendation completed: {len(successful_users)}/{len(self.user_manager.get_all_users())} users")
        self._log_detailed_message('recommendations', 'INFO', f"Round {cycle} recommendation phase statistics", recommendation_stats)


        self._cleanup_gpu_memory()

        return recommendations_data
    
    def _start_vllm_server(self,
                            model_path: str = "/data/teamshare/models/Qwen3-8B",
                            port: int = 8002,
                            cuda_devices: str = "1",
                            ready_timeout: int = 180):
        """Start vllm serve as background process and wait for service to be ready.

        Args:
            model_path:      Model directory path
            port:            vllm listening port
            cuda_devices:    CUDA_VISIBLE_DEVICES environment variable value
            ready_timeout:   Maximum seconds to wait for service to be ready
        """
        import urllib.request


        if self._vllm_process is not None:
            if self._vllm_process.poll() is None:
                print("vllm server is already running, skipping")
                self.logger.info("vllm server is already running, skipping")
                return
            else:
                self._vllm_process = None

        cmd = [
            "vllm", "serve", model_path,
            "--port", str(port)
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        print(f"▶ Starting vllm server: CUDA_VISIBLE_DEVICES={cuda_devices} vllm serve {model_path} --port {port}")
        self.logger.info(f" Starting vllm server: {' '.join(cmd)}")

        self._vllm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )


        health_url = f"http://0.0.0.0:{port}/health"
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"✓ vllm server started successfully (port={port})")
                        self.logger.info(f"vllm server started successfully, port: {port}")
                        return
            except Exception:
                pass


            if self._vllm_process.poll() is not None:
                print(f"✗ vllm server failed to start, return code: {self._vllm_process.returncode}")
                self.logger.error(f"vllm server failed to start, return code: {self._vllm_process.returncode}")
                self._vllm_process = None
                sys.exit(1)

            time.sleep(5)


            print(f"✗ vllm server failed to start within {ready_timeout}s, exiting")
            self.logger.error(f"vllm server startup timeout ({ready_timeout}s)")
        self._stop_vllm_server()
        sys.exit(1)

    def _stop_vllm_server(self):
        """Terminate background vllm process and release GPU resources."""
        import signal

        if self._vllm_process is None:
            return

        if self._vllm_process.poll() is not None:
            self._vllm_process = None
            return

        pid = self._vllm_process.pid
        print(f"■ Stopping vllm server (PID={pid})...")
        self.logger.info(f" Stopping vllm server with PID={pid}")

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            try:
                self._vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._vllm_process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            self.logger.warning(f" Error stopping vllm server: {e}")

        self._vllm_process = None
        print("✓ vllm server stopped")
        self.logger.info("vllm server stopped")

    def _generate_recommendations(self) -> bool:
        """Run recommendation generation script"""
        try:
            print(f"Running recommendation generation: {self.recommend_script}")
            

            if not os.path.exists(self.recommend_script):
                print(f"Error: Recommendation generation script not found: {self.recommend_script}")
                return False
            

            cmd = [
                'bash',
                self.recommend_script,
                str(self.dataset_name),
                str(self.backbone),
                str(self.storage_dir),
                str(self.current_cycle),
                str(self.num_users)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Recommendation generation completed")
                if result.stdout:
                    print(f"Output: {result.stdout.strip()}")
                return True
            else:
                print(f"✗ Recommendation generation failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr.strip()}")
                if result.stdout:
                    print(f"Output: {result.stdout.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Recommendation generation timeout")
            return False
        except Exception as e:
            print(f"✗ Recommendation generation error: {e}")
            if self.enable_logging:
                self.logger.error(f"Recommendation generation failed: {e}", exc_info=True)
            return False

    def _run_interaction_cycle(self, cycle: int, recommendations_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run interaction cycle - process user interactions with multi-threading"""
        print(f"\n--- Round {cycle} (Concurrent users: {self.max_concurrent_users})---")
        self.logger.info(f"Starting Round {cycle} - User interactions")


        completed_users, pending_reflection_users = self._get_completed_users_for_cycle(cycle, self.reflection_interval)


        if pending_reflection_users:
            print(f" Running reflection for {len(pending_reflection_users)} users...")
            for user_id in pending_reflection_users:
                try:
                    user = self.user_manager.get_user(user_id)
                    if user and user.avatar:
                        user.avatar.reflect_on_memory()
                        user.avatar.save_memory()
                        print(f" ✓ User {user_id} reflection completed")
                except Exception as e:
                    print(f" ✗ User {user_id} reflection failed: {e}")


        all_completed_users = completed_users | pending_reflection_users

        interactions_data = []


        if all_completed_users:
            user_memory_dir = self.storage_dir / f"user_memory_{self.dataset_name}"
            for user_id in all_completed_users:
                try:
                    user_num = user_id.split('_')[1]
                    memory_file = user_memory_dir / f"user_{user_num}_memory.json"
                    if not memory_file.exists():
                        continue
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for record in data.get('full_conversation_history', []):
                        if record.get('round') == cycle:
                            selected_item_id = record.get('selected_item_id')
                            interactions_data.append({
                                "user_id": user_id,
                                "cycle": cycle,
                                "evaluations": [{"item_id": selected_item_id, "interact": selected_item_id is not None}] if selected_item_id else [],
                                "selected_items": [{"item_id": selected_item_id, "interact": True}] if selected_item_id else [],
                                "timestamp": time.time()
                            })
                            break
                except Exception as e:
                    self.logger.warning(f" Loading memory for user {user_id} failed: {e}")
            if interactions_data:
                print(f" Loaded memory for {len(interactions_data)} users from Round {cycle} data")


            original_count = len(recommendations_data)
            recommendations_data = [
                rec for rec in recommendations_data
                if rec["user_id"] not in all_completed_users
            ]
            print(f" Skipping {original_count - len(recommendations_data)} completed users, processing {len(recommendations_data)} users")

        if not recommendations_data:
            print(f" All users completed Round {cycle} interactions")
            return interactions_data
        successful_interactions = []
        failed_interactions = []


        global_api_stats = {
            "total_api_calls": 0,
            "total_response_time": 0.0,
            "users_with_api_calls": 0
        }


        def process_user_interaction(user_rec):
            """Process individual user interaction"""
            user_id = user_rec["user_id"]
            recommendations = user_rec["recommendations"]

            try:
                user = self.user_manager.get_user(user_id)
                if not user:
                    error_result = {
                        "user_id": user_id,
                        "reason": "User does not exist",
                        "recommendations_count": len(recommendations),
                        "error": True
                    }
                    print(f"User {user_id} does not exist")
                    self.logger.error(f"User {user_id} does not exist")
                    return error_result


                user.start_new_session(cycle)


                evaluations = user.evaluate_recommendations(recommendations)

                interaction_data = {
                    "user_id": user_id,
                    "cycle": cycle,
                    "recommendations": recommendations,
                    "evaluations": evaluations,
                    "selected_items": [rec for rec in evaluations if rec.get('interact')],
                    "timestamp": time.time(),
                    "error": False
                }

                selected_count = len(interaction_data["selected_items"])


                interaction_details = {
                    "user_id": user_id,
                    "cycle": cycle,
                    "total_recommendations": len(evaluations),
                    "selected_items_count": selected_count,
                    "acceptance_rate": selected_count / len(evaluations) * 100 if evaluations else 0,
                    "total_interactions_after": user.total_interactions,
                    "selected_items": [
                        {
                            "item_id": item.get('item_id'),
                            "rating": item.get('user_rating'),
                        } for item in interaction_data["selected_items"][:3]
                    ]
                }

                print(f"User {user_id}: Evaluated {len(evaluations)} items, selected {selected_count}")
                self.logger.debug(f"User {user_id} interaction completed: {selected_count}/{len(evaluations)} items selected")
                self._log_detailed_message('user_interactions', 'INFO', f"User interaction details", interaction_details)


                user_api_stats = user.avatar.get_current_round_response_stats()

                return {
                    "interaction_data": interaction_data,
                    "interaction_details": interaction_details,
                    "user_api_stats": user_api_stats,
                    "error": False
                }

            except Exception as e:
                error_result = {
                    "user_id": user_id,
                    "error": str(e),
                    "recommendations_count": len(recommendations),
                    "error": True
                }
                print(f"User {user_id} interaction failed: {e}")
                self.logger.error(f"User {user_id} interaction failed: {e}")
                self._log_detailed_message('errors', 'ERROR', f"User interaction failed", {
                    "user_id": user_id,
                    "cycle": cycle,
                    "error": str(e),
                    "recommendations_count": len(recommendations)
                })
                return error_result


        with ThreadPoolExecutor(max_workers=self.max_concurrent_users) as executor:

            future_to_user_rec = {executor.submit(process_user_interaction, user_rec): user_rec for user_rec in recommendations_data}


            for future in as_completed(future_to_user_rec):
                user_rec = future_to_user_rec[future]
                user_id = user_rec["user_id"]

                try:
                    result = future.result()

                    if result.get("error", False):
                        failed_interactions.append(result)
                    else:
                        interactions_data.append(result["interaction_data"])
                        successful_interactions.append(result["interaction_details"])

                        user_api_stats = result.get("user_api_stats", {})
                        if user_api_stats.get("count", 0) > 0:
                            global_api_stats["total_api_calls"] += user_api_stats["count"]
                            global_api_stats["total_response_time"] += user_api_stats["total_time"]
                            global_api_stats["users_with_api_calls"] += 1

                except Exception as e:
                    print(f"Processing user {user_id} result failed: {e}")
                    self.logger.error(f"Processing user {user_id} result failed: {e}")
                    failed_interactions.append({
                        "user_id": user_id,
                        "error": str(e),
                        "recommendations_count": len(user_rec.get("recommendations", [])),
                        "error": True
                    })


        total_selected = sum(len(data["selected_items"]) for data in interactions_data)
        total_recommendations = sum(len(data["evaluations"]) for data in interactions_data)


        global_api_stats["avg_response_time"] = (
            global_api_stats["total_response_time"] / global_api_stats["total_api_calls"]
            if global_api_stats["total_api_calls"] > 0 else 0
        )


        interaction_stats = {
            "cycle": cycle,
            "total_users_interacted": len(successful_interactions),
            "failed_interactions": len(failed_interactions),
            "total_recommendations_evaluated": total_recommendations,
            "total_items_selected": total_selected,
            "overall_acceptance_rate": total_selected / total_recommendations * 100 if total_recommendations > 0 else 0,
            "avg_selections_per_user": total_selected / len(successful_interactions) if successful_interactions else 0,
            "concurrent_processing": True,
            "max_concurrent_users": self.max_concurrent_users,
            "global_api_stats": global_api_stats,
            "successful_interaction_details": successful_interactions,
            "failed_interaction_details": failed_interactions
        }

        print(f"✓ Interaction cycle completed, {total_selected} items selected (processing {len(recommendations_data)} users)")
        if global_api_stats["total_api_calls"] > 0:
            print(f"✓ API stats: {global_api_stats['users_with_api_calls']} users made {global_api_stats['total_api_calls']} API calls, total time: {global_api_stats['total_response_time']:.3f}s, avg: {global_api_stats['avg_response_time']:.3f}s")
            self.logger.info(f"Round {cycle} interaction completed: {len(successful_interactions)} users, {total_selected} items selected")
        self._log_detailed_message('user_interactions', 'INFO', f"Round {cycle} interaction phase statistics", interaction_stats)

        return interactions_data

    def _ensure_cycle_data_exists(self, cycle: int):
        """Ensure current cycle data file exists, generate if not

        Called at the start of each round, generates current round data from previous round data + previous round interactions
        """
        data_dir = self.storage_dir / "data"
        current_data_file = data_dir / f"{self.dataset_name}-{cycle}.txt"



        is_resume_start = (cycle == self.start_cycle and cycle > 1)
        if current_data_file.exists() and not is_resume_start:
            print(f" Data file already exists: {current_data_file}")
            return
        if current_data_file.exists() and is_resume_start:
            print(f" Resuming from start cycle, using existing data file: {current_data_file}")

            print(f"\n--- Generating Round {cycle} data file ---")

        if cycle == 1:

            src_file = Path(f"./recommenders/data/{self.dataset_name}/{self.dataset_name}.txt")
            if src_file.exists():
                shutil.copy2(str(src_file), str(current_data_file))
                print(f" Copied initial data: {current_data_file}")
            else:
                print(f" Error: Initial data file does not exist: {src_file}")
            return


        prev_data_file = data_dir / f"{self.dataset_name}-{cycle-1}.txt"
        if not prev_data_file.exists():
            print(f" Error: Previous round data file does not exist: {prev_data_file}")
            return


        with open(prev_data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        user_lines = {}
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if parts:
                user_id = int(parts[0])
                user_lines[user_id] = i


        user_memory_dir = self.storage_dir / f"user_memory_{self.dataset_name}"
        prev_cycle = cycle - 1
        new_interactions_count = 0

        if user_memory_dir.exists():
            for memory_file in user_memory_dir.glob("user_*_memory.json"):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    user_id_str = data.get('user_id', '')
                    user_num = int(user_id_str.split('_')[1])

                    for record in data.get('full_conversation_history', []):
                        if record.get('round') == prev_cycle:
                            selected_item_id = record.get('selected_item_id')
                            if selected_item_id is not None:
                                if user_num in user_lines:
                                    line_idx = user_lines[user_num]
                                    lines[line_idx] = lines[line_idx].strip() + f" {selected_item_id}\n"
                                else:
                                    lines.append(f"{user_num} {selected_item_id}\n")
                                    user_lines[user_num] = len(lines) - 1
                                new_interactions_count += 1
                            break
                except Exception as e:
                    continue


        with open(current_data_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

            print(f" Generated data file: {current_data_file}")
            print(f" Added {new_interactions_count} new interactions")

    def _run_data_update_cycle(self, cycle: int, interactions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run data update cycle"""
        print(f"\n--- Round {cycle} data update ---")
        self.logger.info(f"Starting Round {cycle} data update")


        has_new_interactions = any(
            interaction.get("selected_items")
            for interaction in interactions_data
        )

        if not has_new_interactions:
            print(f" No new interactions to update, skipping data update")
            self.logger.info(f"Round {cycle} no new interactions, skipping data update")
            return {
                "cycle": cycle,
                "update_success": False,
                "new_interactions": 0,
                "status": "no_new_data",
                "timestamp": time.time()
            }

        try:

            pre_update_stats = self.data_manager.get_data_statistics()
            self.data_manager.data_file = self.storage_dir / "data" / f"{self.dataset_name}-{cycle+1}.txt"

            self._append_interactions_to_dataset(interactions_data)
            
            update_stats = self.incremental_generator.generate_incremental_data(
                cycle_number=cycle,
                min_interactions_threshold=1
            )

            data_stats = self.data_manager.get_data_statistics()

            combined_stats = {
                "cycle": cycle,
                "update_success": update_stats.get("status") not in ["no_new_data", "error"],
                "new_interactions": update_stats.get("new_interactions", 0),
                "data_stats": data_stats,
                "pre_update_stats": pre_update_stats,
                "status": update_stats.get("status", "unknown"),
                "timestamp": time.time()
            }

            if update_stats.get("status") == "error":
                combined_stats["error"] = update_stats.get("error")
                print(f"✗ Data update failed: {update_stats.get('error')}")
                self.logger.error(f"Round {cycle} data update failed: {update_stats.get('error')}")
                self._log_detailed_message('data_updates', 'ERROR', f"Data update failed", combined_stats)
            else:
                print(f"✓ Data update completed: {combined_stats['new_interactions']} new interactions")
                self.logger.info(f"Round {cycle} data update completed: {combined_stats['new_interactions']} new interactions")
                self._log_detailed_message('data_updates', 'INFO', f"Data update statistics", combined_stats)
            
            return combined_stats

        except Exception as e:
            print(f"Data update failed: {e}")
            self.logger.error(f"Round {cycle} data update error: {e}")
            error_stats = {
                "cycle": cycle,
                "error": str(e),
                "timestamp": time.time()
            }
            self._log_detailed_message('errors', 'ERROR', f"Data update exception", error_stats)
            return {"error": str(e)}

    def _run_training_cycle(self, cycle: int) -> Optional[Dict[str, Any]]:
        """Run model training cycle (simplified implementation)"""
        print(f"\n--- Round {cycle} model training ---")
        self.logger.info(f"Starting Round {cycle} model training")

        try:
            training_start_time = time.time()
            
            training_stats = {
                "cycle": cycle,
                "model_updated": True,
                "training_time": 0.1,
                "training_status": "simplified_implementation",
                "timestamp": time.time()
            }

            training_duration = time.time() - training_start_time
            training_stats["actual_training_time"] = training_duration

            print("✓ Model training completed (simplified implementation)")
            self.logger.info(f"Round {cycle} model training completed, duration: {training_duration:.3f}s")
            self._log_detailed_message('training', 'INFO', f"Model training statistics", training_stats)
            
            return training_stats
            
        except Exception as e:
            print(f"Model training failed: {e}")
            self.logger.error(f"Round {cycle} model training failed: {e}")
            error_stats = {
                "cycle": cycle,
                "error": str(e),
                "timestamp": time.time()
            }
            self._log_detailed_message('errors', 'ERROR', f"Model training failed", error_stats)
            return {"error": str(e)}

    def _save_cycle_checkpoint(self, cycle: int, cycle_metrics: Dict[str, Any]):
        """Save cycle checkpoint"""
        checkpoint_file = self.storage_dir / f"checkpoint_cycle_{cycle}.json"

        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "cycle": cycle,
            "metrics": cycle_metrics,
            "system_stats": self._get_system_stats(),
            "timestamp": time.time()
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        user_stats = self.user_manager.get_user_stats_summary()
        
        system_stats = {
            "total_users": len(self.user_manager.get_all_users()),
            "user_stats": user_stats,
            "current_cycle": self.current_cycle,
            "dataset": self.dataset_name,
            "experiment_id": self.experiment_id,
            "timestamp": time.time()
        }
        

        self._log_detailed_message('system_stats', 'INFO', f"System status statistics - Round {self.current_cycle}", system_stats)
        
        return system_stats

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory
        
        Called after each recommendation generation round to release accumulated GPU memory.
        This is crucial to avoid CUDA out of memory errors during long runs.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.debug(f"GPU memory cleanup completed - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup failed: {e}")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final report"""
        total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
        total_interactions = sum(cycle["new_interactions"] for cycle in self.performance_metrics["cycle_metrics"])

        report = {
            "experiment_id": self.experiment_id,
            "configuration": {
                "dataset": self.dataset_name,
                "num_users": self.num_users,
                "items_per_cycle": self.items_per_cycle,
                "max_cycles": self.max_cycles
            },
            "performance": {
                "total_time": round(total_time, 2),
                "cycles_completed": self.performance_metrics["cycles_completed"],
                "total_interactions": total_interactions,
                "avg_cycle_time": round(total_time / self.max_cycles, 2),
                "interactions_per_cycle": round(total_interactions / self.max_cycles, 2)
            },
            "cycle_details": self.performance_metrics["cycle_metrics"],
            "final_system_stats": self._get_system_stats(),
            "data_stats": self.data_manager.get_data_statistics(),
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        report_file = self.storage_dir / f"final_report_{self.experiment_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"✓ Final report saved to: {report_file}")
            self.logger.info(f"Final report generated: {report_file}")
        

        self._log_detailed_message('system_stats', 'INFO', f"Final experiment report", report)
        
        return report

    def save_system_state(self):
        """Save system state"""
        state_file = self.storage_dir / f"system_state_{self.experiment_id}.json"

        state = {
            "experiment_id": self.experiment_id,
            "current_cycle": self.current_cycle,
            "performance_metrics": self.performance_metrics,
            "system_config": {
                "dataset": self.dataset_name,
                "num_users": self.num_users,
                "items_per_cycle": self.items_per_cycle,
                "max_cycles": self.max_cycles
            },
            "saved_at": time.time()
        }

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

            print(f"✓ System state saved to: {state_file}")

    def load_system_state(self, state_file: str):
        """Load system state"""
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.experiment_id = state["experiment_id"]
        self.current_cycle = state["current_cycle"]
        self.performance_metrics = state["performance_metrics"]

        print(f"✓ System state loaded from: {state_file}")

    def _get_completed_users_for_cycle(self, cycle: int, reflection_interval: int = 5) -> tuple:
        """Get set of user IDs that have completed interactions for specified round, and users needing reflection

        Determine which users have completed interactions for this round by checking interaction_count in user memory files
        Determine if reflection is completed by checking time difference

        Args:
            cycle: Round number
            reflection_interval: Reflection interval

        Returns:
            (Set of fully completed user IDs, Set of user IDs needing reflection)
        """
        from datetime import datetime

        completed_users = set()
        pending_reflection_users = set()
        user_memory_dir = self.storage_dir / f"user_memory_{self.dataset_name}"

        if not user_memory_dir.exists():
            return completed_users, pending_reflection_users


        for memory_file in user_memory_dir.glob("user_*_memory.json"):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                user_id = data.get("user_id")
                interaction_count = data.get("interaction_count", 0)


                if interaction_count < cycle:
                    continue


                needs_reflection = (reflection_interval > 0 and
                                   interaction_count % reflection_interval == 0)

                if needs_reflection:

                    last_updated = data.get("last_updated", "")
                    full_history = data.get("full_conversation_history", [])

                    if full_history and last_updated:
                        last_record_time = full_history[-1].get("timestamp", "")
                        if last_record_time:
                            try:
                                updated_dt = datetime.fromisoformat(last_updated)
                                record_dt = datetime.fromisoformat(last_record_time)
                                time_diff = (updated_dt - record_dt).total_seconds()

                                if time_diff < 1.0:
                                    pending_reflection_users.add(user_id)
                                else:
                                    completed_users.add(user_id)
                            except Exception:
                                pending_reflection_users.add(user_id)   
                        else:
                            pending_reflection_users.add(user_id)
                    else:
                        pending_reflection_users.add(user_id)
                else:
                    completed_users.add(user_id)

            except Exception as e:
                pass

        if completed_users:
            print(f"✓ {len(completed_users)} users completed Round {cycle} interactions")
        if pending_reflection_users:
            print(f"⚠ {len(pending_reflection_users)} users need reflection for Round {cycle}")

        return completed_users, pending_reflection_users

    def _load_recommendations_from_file(self) -> Dict[int, List[int]]:
        """Load recommendations from file
        
        Supports the following formats:
        1. Simple format: Each line "user_id item1 item2 item3 ..."
        2. JSON format: {"user_id": [item1, item2, item3, ...], ...}
        
        Returns:
            Mapping from user ID to recommendation item list
        """
        recommendations = {}
        
        if not os.path.exists(self.recommendation_file_path):
            print(f"Recommendation file does not exist: {self.recommendation_file_path}")
            return recommendations
        
        try:

            if self.recommendation_file_path.endswith('.json'):
                with open(self.recommendation_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id_str, items in data.items():
                        try:
                            user_id = int(user_id_str)
                            recommendations[user_id] = [int(item) for item in items]
                        except (ValueError, TypeError):
                            continue
            else:

                with open(self.recommendation_file_path, 'r', encoding='utf-8') as f:
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
                                recommendations[user_id] = item_ids
                        except (ValueError, IndexError):
                            continue
            
                            print(f"✓ fileload {len(recommendations)} usersrecommendationresult")
            return recommendations
            
        except Exception as e:
            print(f"loadrecommendationfilefailed: {e}")
            return {}

    def _enrich_recommendations_with_item_info(self, recommendations: List[int]) -> Optional[List[Dict[str, Any]]]:
        """Enrich recommendations with item information using item retriever
        
        Args:
            recommendations: List of item IDs
            
        Returns:
            List of dictionaries containing complete item information
        """
        if not self.item_retriever:

            raise Exception("")

        try:
            enriched_recs = []
            for item_id in recommendations:

                item_info = self.item_retriever.get_item_info(item_id)
                
                if item_info:

                    enriched_rec = {
                        'item_id': item_id,
                        'id': item_id,
                        **item_info
                    }
                    enriched_recs.append(enriched_rec)
                else:

                    raise Exception(f"found: {item_id}")
                    
            return enriched_recs
            
        except Exception as e:
            print(f"recommendation: {e}")

            raise Exception(f"recommendation: {e}")

    def _append_interactions_to_dataset(self, interactions_data: List[Dict[str, Any]]):
        """Append new interaction data to dataset file"""

        dataset_file = self.storage_dir / "data" / f"{self.dataset_name}-{self.current_cycle}.txt"
        
        try:

            user_lines = {}
            with open(dataset_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if parts:
                    user_id = int(parts[0])
                    user_lines[user_id] = i
            

            for interaction_data in interactions_data:
                user_id_str = interaction_data["user_id"]

                user_num = int(user_id_str.split('_')[1])
                
                selected_items = interaction_data.get("selected_items", [])
                if not selected_items:
                    continue
                    

                item_ids = []
                for item in selected_items:

                    if isinstance(item, dict):
                        item_id = item.get('item_id') or item.get('id')
                    else:
                        item_id = item
                    
                    if item_id is not None:
                        item_ids.append(str(item_id))
                
                if not item_ids:
                    continue
                    

                if user_num in user_lines:

                    line_idx = user_lines[user_num]
                    current_line = lines[line_idx].strip()
                    new_line = current_line + " " + " ".join(item_ids) + "\n"
                    lines[line_idx] = new_line
                    print(f"users {user_num}: {len(item_ids)} ")
                else:

                    new_line = f"{user_num} " + " ".join(item_ids) + "\n"
                    lines.append(new_line)
                    user_lines[user_num] = len(lines) - 1
                    print(f"users {user_num}: , {len(item_ids)} ")
            

            next_cycle_dataset_file = self.storage_dir / "data" / f"{self.dataset_name}-{self.current_cycle+1}.txt"
            with open(next_cycle_dataset_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
                print(f"✓ data {next_cycle_dataset_file}")
            
        except Exception as e:
            print(f"datafailed: {e}")
            if self.enable_logging:
                self.logger.error(f"datafailed: {e}", exc_info=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Closed-loop recommendation system")
    parser.add_argument('--dataset', type=str, default='Toys_and_Games', help='Dataset name')
    parser.add_argument('--backbone', type=str, default='SASRec', help='Model backbone')
    parser.add_argument('--num_users', type=int, default=5, help='User count')
    parser.add_argument('--items_per_cycle', type=int, default=5, help='Number of recommendation items per cycle')
    parser.add_argument('--max_cycles', type=int, default=1, help='Maximum number of cycles')
    parser.add_argument('--start_cycle', type=int, default=1, help='Starting cycle number')
    parser.add_argument('--reflection_interval', type=int, default=5, help='Reflection interval (every N cycles, 0 to disable)')
    parser.add_argument('--storage_dir', type=str, default='./simulation_storage', help='Storage directory')
    parser.add_argument('--recommendation_file', type=str, help='Recommendation result file path')
    parser.add_argument('--recommend_script', type=str, help='Recommendation generation script path')
    parser.add_argument('--no_training', action='store_true', help='disablemodeltraining')
    parser.add_argument('--quiet', action='store_true', help='')
    parser.add_argument('--max_concurrent_users', type=int, default=5, help='User count')

    args = parser.parse_args()

    system = ClosedLoopRecommendationSystem(
        dataset_name=args.dataset,
        backbone=args.backbone,
        num_users=args.num_users,
        items_per_cycle=args.items_per_cycle,
        max_cycles=args.max_cycles,
        storage_dir=args.storage_dir,
        recommendation_file=args.recommendation_file,
        recommend_script=args.recommend_script,
        enable_training=not args.no_training,
        enable_logging=not args.quiet,
        max_concurrent_users=args.max_concurrent_users,
        start_cycle=args.start_cycle,
        reflection_interval=args.reflection_interval
    )

    try:
        if not system.initialize_system():
            print("failed")
            return

        results = system.run_closed_loop_simulation()
        system.save_system_state()

        print("=== ===" )
        print(f"ID: {results['experiment_id']}")
        print(f": {results['performance']['total_time']:.2f}s")
        print(f"completed: {results['performance']['cycles_completed']}")
        print(f": {results['performance']['total_interactions']}")
        print(f": {results['performance']['interactions_per_cycle']:.2f}")

    except KeyboardInterrupt:
        print("\n, Runningsave...")
        system.save_system_state()
        print("save")

    except Exception as e:
        print(f"error: {e}")
        system.save_system_state()
        raise


if __name__ == '__main__':
    main()
