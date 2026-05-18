"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import trange

# HuggingFace Sentence-T5 IDs; override local root via SENTENCE_T5_MODEL_ROOT
SENTENCE_T5_HF_IDS = {
    "sentence-t5-base": "sentence-transformers/sentence-t5-base",
    "sentence-t5-large": "sentence-transformers/sentence-t5-large",
    "sentence-t5-xl": "sentence-transformers/sentence-t5-xl",
    "sentence-t5-xxl": "sentence-transformers/sentence-t5-xxl",
}


def _sentence_t5_model_id(content_model: str) -> str:
    """Resolve sentence-t5 config name to a HuggingFace or local model path."""
    if content_model in SENTENCE_T5_HF_IDS:
        return SENTENCE_T5_HF_IDS[content_model]
    model_root = os.environ.get("SENTENCE_T5_MODEL_ROOT")
    if model_root:
        local_path = os.path.join(model_root, content_model)
        if os.path.isdir(local_path):
            return local_path
    if os.path.isabs(content_model) and os.path.isdir(content_model):
        return content_model
    return f"sentence-transformers/{content_model}"


def process_embeddings(
    config,
    device,
    id2meta_file,
    embedding_save_path,
):
    item_2_text = json.loads(open(id2meta_file).read())
    item_id_2_text = {}
    for k, v in item_2_text.items():
        item_id_2_text[int(k)] = v

    if os.path.exists(embedding_save_path):
        item_embedding = torch.load(embedding_save_path, weights_only=False, map_location=device)

    else:
        print("Embeddings not found, generating embeddings...")
        content_model = config["dataset"]["content_model"]
        if "sentence-t5" in content_model:
            with torch.no_grad():
                from sentence_transformers import SentenceTransformer
                text_embedding_model = SentenceTransformer(
                    _sentence_t5_model_id(content_model), device=device
                )
                sorted_text = [value for key, value in sorted(item_id_2_text.items())]
            bs = 512 if content_model == "sentence-t5-base" else 4
            # embedding is generated based on the sorted text
            with torch.no_grad():
                embeddings = text_embedding_model.encode(
                    sorted_text,
                    convert_to_numpy=True,
                    batch_size=bs,
                    show_progress_bar=True,
                )
            # embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # it is by default have norm 1, this is to just ensure
            with open(embedding_save_path, "wb") as f:
                pickle.dump(embeddings, f)
        elif "bert" in content_model:
            from transformers import BertModel, BertTokenizer

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").to(device)
            bs = 32
            # embedding is generated based on the sorted text
            sorted_text = [value for key, value in sorted(item_id_2_text.items())]
            total_length = len(sorted_text)
            embedding_list = []
            for i in trange(total_length // bs + 1):
                input_text = sorted_text[i * bs : (i + 1) * bs]
                with torch.no_grad():
                    inputs = tokenizer(
                        input_text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(device)
                    output = model(**inputs, return_dict=True)  # [bs, 1024]
                    embeddings = output.last_hidden_state[:, 0]
                    embedding_list.append(embeddings)

            embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
            with open(embedding_save_path, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            raise NotImplementedError

        embeddings = StandardScaler().fit_transform(embeddings)
        item_embedding = torch.Tensor(embeddings).to(device)
        torch.save(item_embedding, embedding_save_path)
    return item_embedding


def process_data_split(
    config,
    data_file,
    id2meta_file,
    is_steam=False,
):

    max_items_per_seq = config["dataset"]["max_items_per_seq"]
    print(f" - : {max_items_per_seq}")
    item_2_text = json.loads(open(id2meta_file).read())
    item_id_2_text = {}
    for k, v in item_2_text.items():
        item_id_2_text[int(k)] = v

    user_sequence = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            user_sequence.append(
                [int(x) for x in line.split(" ")[1:]]
            )  # this sequence only contains item ids

    # cut the user_sequence first before filtering out unseen val and test
    user_sequence = [
        seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:]
        for seq in user_sequence
    ]

    if is_steam:  # subsample the data for steam
        user_sequence = user_sequence[::7]

    val_items = []
    test_items = []
    all_items = []
    train_items = []
    for i in range(len(user_sequence)):
        test_items.append(user_sequence[i][-1])
        if len(user_sequence[i]) > 1:
            val_items.append(user_sequence[i][-2])
        if len(user_sequence[i]) > 2:
            train_items.extend(user_sequence[i][:-2])
        all_items.extend(user_sequence[i][:])

    test_items = np.unique(test_items)
    val_items = np.unique(val_items)
    train_items = np.unique(train_items)
    all_items = np.unique(all_items)
    item_to_ind = {
        int(key): i for i, (key, _) in enumerate(sorted(item_id_2_text.items()))
    }  # this will just be {k: k-1}

    unseen_test = np.setdiff1d(test_items, train_items)
    unseen_val = np.setdiff1d(val_items, train_items)
    train_items2 = all_items[
        ~np.isin(all_items, np.concatenate((unseen_val, unseen_test)))
    ]
    assert (train_items == train_items2).sum() == train_items.shape[
        0
    ], "Something is wrong in identifying the unique training items, causing discrepancy in the different ways of filtering the train items. Please check."
    train_inds = [item_to_ind[item] for item in train_items]
    assert (np.array(train_inds) == train_items - 1).all()

    saved_semantic_id_split = {}
    saved_semantic_id_split["unseen_val"] = unseen_val
    saved_semantic_id_split["unseen_test"] = unseen_test
    saved_semantic_id_split["seen"] = train_items

    return saved_semantic_id_split, user_sequence
