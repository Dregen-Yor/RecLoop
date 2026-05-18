# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import traceback

import hydra

import torch
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training import train_tiger

from utils import cf_embedding_path, resolve_generativerec_path, set_seed


class set_dir:
    def __init__(self, config):
        raw_data_path = config["dataset"].get("raw_data_path", "../data/Amazon2014/")
        processed_root = config["dataset"].get(
            "processed_data_path", "./ID_generation/preprocessing/processed/"
        )
        self.directory = resolve_generativerec_path(raw_data_path)
        self.directory_processed = os.path.join(
            resolve_generativerec_path(processed_root), config["dataset"]["name"]
        )
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        dataset_name = config["dataset"]["name"]
        id_filename = f"{dataset_name}_{config['dataset']['content_model']}"

        if config["test_method"] in ["tiger", "liger"]:
            self.rqvae_save_dir = resolve_generativerec_path("./ID_generation/ID/")
            os.makedirs(self.rqvae_save_dir, exist_ok=True)
            self.id_save_location = os.path.join(
                self.rqvae_save_dir, id_filename + f"_{config['seed']}.pkl"
            )

        self.embedding_save_name = f"_{config['dataset']['content_model']}"
        if config["dataset"].get("text_emb", False):
            self.embedding_save_path = os.path.join(
                self.directory_processed, f"{id_filename}_embeddings.pt"
            )
        else:
            self.embedding_save_path = cf_embedding_path(
                config["dataset"]["cf_model"], dataset_name
            )

        self.result_save_dir = f"./results/{config['test_method']}/"
        os.makedirs(self.result_save_dir, exist_ok=True)

    def set_config(self, config):
        config["dataset"]["raw_data_path"] = self.directory
        config["dataset"]["processed_data_path"] = self.directory_processed
        config["output_path"] = os.path.join(
            self.result_save_dir,
            f"{config['dataset']['type']}_{config['dataset']['name']}",
            f"{config['experiment_id']}_seed_{config['seed']}",
        )
        os.makedirs(config["output_path"], exist_ok=True)
        return config


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    print("\n" + "=" * 80)
    print("🚀 Start LIGER training")
    print("=" * 80)
    
    # print(config)
    print(f"\n[ 1/7] random seed...")
    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f" ✓ : {device}")
    set_seed(config["seed"])
    print(f" ✓ random seed: {config['seed']}")

    print(f"\n[ 2/7] directory...")
    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    config["logging"]["project"] = "liger"
    is_steam = config["dataset"]["type"] == "steam"
    print(f" ✓ outputpath: {config['output_path']}")
    print(f" ✓ dataset: {'Steam' if is_steam else 'Amazon'}")

    try:
        print(f"\n[ 3/7] dataprocessing...")

        dataset_name = config["dataset"]["name"]
        features_needed = config["dataset"]["features_needed"]
        prompt_format = config["dataset"]["prompt_format"]
        processed_data_path = config["dataset"]["processed_data_path"]
        print(f" - dataset: {dataset_name}")
        print(f" - : {features_needed}")
        print(f" - : {prompt_format}")
        
        features_used = "_".join(features_needed)
        data_file = os.path.join(processed_data_path, f"{dataset_name}.txt")
        id2meta_file = os.path.join(
            processed_data_path,
            f"{dataset_name}_{features_used}_{prompt_format}_id2meta.json",
        )
        item2attribute_file = os.path.join(
            processed_data_path, f"{dataset_name}_item2attributes.json"
        )
        print(f"currentdirectory: {os.getcwd()}")
        print(f" - datafile: {data_file}")
        print(f" - datafile: {id2meta_file}")
        print(f" - Item attributesfile: {item2attribute_file}")

        files_exist = (
            os.path.exists(data_file) and 
            os.path.exists(id2meta_file) and 
            os.path.exists(item2attribute_file)
        )
        
        if files_exist:
            print(" ✓ foundProcesseddatafile, dataprocessing")
            print(f"    - Interaction sequence: {data_file}")
            print(f" - data: {id2meta_file}")
            print(f"    - Item attributes: {item2attribute_file}")
        else:
            print(" ⚠ not foundProcesseddatafile, Startdataprocessing...")
            data_file, id2meta_file, item2attribute_file = preprocessing(config["dataset"])
            print(" ✓ dataprocessingcompleted")
        
        # id2meta_file: the file that save item_id to meta info, we will later use it for sentence T5 embedding generation
        # data_file: the file that save the user-item interactions.

        train_config = {
            **config["dataset"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }
        method_config = {
            **config["method"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }

        # load id split
        print(f"\n[ 4/7] loaddata...")
        print(f" - datafile: {data_file}")
        print(f"  - Reading metadata: {id2meta_file}")
        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )
        print(f" ✓ datacompleted")
        print(f"    - User count: {len(user_sequence)}")
        print(f" - Seen: {len(id_split['seen'])}")
        print(f" - Unseen Val: {len(id_split['unseen_val'])}")
        print(f" - Unseen Test: {len(id_split['unseen_test'])}")
        # load item embedding
        print(f"\n[ 5/7] load...")
        print(f" - savepath: {PATH_CONFIG.embedding_save_path}")
        if config["dataset"]["text_emb"] is True:
            item_embedding = process_embeddings(
                config, device, id2meta_file, PATH_CONFIG.embedding_save_path
            )
        else:
            item_embedding = torch.load(
                PATH_CONFIG.embedding_save_path, weights_only=False
            ).to(device)

        print(f" ✓ Loading completed")
        print(f" - : {item_embedding.shape}")

        print(f"\n[ 6/7] trainingID (RQ-VAE)...")
        print(f" - IDsave: {PATH_CONFIG.id_save_location}")
        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
        )
        print(f" ✓ IDTraining completed")

        print(f"\n[ 7/7] training TIGER model...")
        print(f" - outputpath: {config['output_path']}")
        train_tiger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            item_embedding,
            PATH_CONFIG.id_save_location,
            device=device,
        )
        print(f" ✓ TIGER modelTraining completed")

        print("\n" + "=" * 80)
        print("🎉 completed！trainingend")
        print("=" * 80 + "\n")

    except BaseException as e:
        print("\n" + "=" * 80)
        print(f"❌ error: training")
        print("=" * 80)
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
