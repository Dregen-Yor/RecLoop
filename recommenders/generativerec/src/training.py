"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from transformers import T5Config
from transformers.optimization import get_scheduler

from utils import CustomDataset, get_lr, setup_logging

from .evaluation import (
    evaluate,
    evaluate_dense_ids,
    evaluate_dense_sids,
    generate_then_dense,
    get_target_embed,
    model_forward,
)
from .load_data import load_data
from .tiger import TIGER


def evaluate_helper(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[5, 10, 20],  # recall@k, k list
    RETRIEVE_KEY=[20, 40, 60, 80, 100],  # retrieve then rank
):
    """
    :param KEYS: the keys for the recall@k
    :param RETRIEVE_KEY: the keys for the retrieve then rank, only used for liger
    """

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate(
                model,
                dataloader,
                all_semantic_ids,
                device,
                method_config=method_config,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )

        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:  # during training, do selected eval
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_5 = logs[f"dense_in_{keyword}/NDCG@5"]
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        ndcg_at_20 = logs[f"dense_in_{keyword}/NDCG@20"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_5 = logs[f"dense_in_{keyword}/NDCG@5"]
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
            ndcg_at_20 = logs[f"dense_in_{keyword}/NDCG@20"]
        else:
            ndcg_at_5 = logs[f"genret_in_{keyword}/NDCG@5"]
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]
            ndcg_at_20 = logs[f"genret_in_{keyword}/NDCG@20"]

    ndcg_metrics = {
        "ndcg@5": ndcg_at_5,
        "ndcg@10": ndcg_at_10,
        "ndcg@20": ndcg_at_20,
    }
    
    return logs, ndcg_metrics


def train_epoch(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch+1}")
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)
    

    total_loss = 0
    total_sid_loss = 0
    total_embedding_loss = 0
    num_batches = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        outputs, _ = model_forward(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Calculating the loss
        loss = 0
        hard_loss = outputs.loss
        loss += hard_loss * method_config["sid_loss_weight"]

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = (
                model.predicted_embedding
            )  # [batch_size, (n_tokens), n_embd]
            _, logits = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]  # [batch_size, n_code]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            # do not update the embedding for the unseen items
            logits[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
        

        total_loss += loss.item()
        total_sid_loss += hard_loss.item()
        if isinstance(embedding_loss, torch.Tensor):
            total_embedding_loss += embedding_loss.item()
        num_batches += 1
        

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })
        progress_bar.update(1)

    progress_bar.close()
    

    avg_loss = total_loss / num_batches
    avg_sid_loss = total_sid_loss / num_batches
    avg_embedding_loss = total_embedding_loss / num_batches if num_batches > 0 else 0
    print(f"\n  Epoch {epoch+1} - Loss: {avg_loss:.4f} | LR: {get_lr(optimizer):.6f}")

    logs = {
        "train/loss": loss.item(),
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
    }
    logs["train/sid_loss"] = hard_loss
    logs["train/embedding_loss"] = embedding_loss

    writer.log(logs)

    return model


def train_tiger(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
):

    output_path = config["output_path"]
    codebook_size = config["RQ-VAE"]["code_book_size"]
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    ######### NOW RE-WRITE CONFIGS ###########
    config = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        item2sid,
    ) = load_data(
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_size=codebook_size,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )  # [n_items, n_code]
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )  # [n_items, n_code]

    # deal with the output embedding
    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    # Check the fourth semantic ID size
    last_codebook_size = max(max_last_semantic_ids, codebook_size)
    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
        )  # by default this is 3026
        # 2000 for user
    else:
        this_vocab_size = codebook_size * n_semantic_codebook + last_codebook_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    trainer_config = config["trainer"]
    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Initialize the model with the custom configuration
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})

    total_epochs = int(np.ceil(total_steps / len(train_dataloader)))
    print(f"\n{'='*80}")
    print(f"TIGER model training configuration:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Total training steps: {total_steps}")
    print(f"  - Steps per epoch: {len(train_dataloader)}")
    print(f"  - Total epochs: {total_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {trainer_config['lr']}")
    print(f"  - Evaluation frequency: every {trainer_config['eval_frequence']} epochs")
    print(f"{'='*80}\n")

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        # then this is the liger method
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        # then this could be TIGER or dense method only
        # RETRIEVE_KEY needs to be at least max(KEYS) for evaluation
        RETRIEVE_KEY = [20]  # Changed from [10] to [20] to support @20 evaluation

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    optimizer = AdamW(
        model.parameters(),
        lr=trainer_config["lr"],
        weight_decay=trainer_config["weight_decay"],
    )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    scheduler = None
    if trainer_config["scheduler"] != "none":
        scheduler = get_scheduler(
            name=trainer_config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=trainer_config["warmup_steps"],
            num_training_steps=total_steps,
        )

    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )  # NOTE: change to cpu if OOM
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"\n✓ Resumed training from checkpoint:")
        print(f"  - Checkpoint path: {state_path}")
        print(f"  - Starting epoch: {start_epoch + 1}")
        print(f"  - Current best NDCG@10: {best_ndcg_10:.4f} (Epoch {best_epoch + 1})")
        print(f"  - Global steps: {global_step}\n")

    print(f"{'='*80}")
    print(f"Starting training loop...")
    print(f"{'='*80}\n")
    
    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        print(f"\n{'─'*80}")
        print(f"📊 Epoch [{epoch+1}/{total_epochs}]")
        print(f"{'─'*80}")
        
        model = train_epoch(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            print(f"\n  🔍 Running validation evaluation...")

            logs, ndcg_metrics = evaluate_helper(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            logs["train/step"] = global_step
            

            ndcg_at_5 = ndcg_metrics["ndcg@5"]
            ndcg_at_10 = ndcg_metrics["ndcg@10"]
            ndcg_at_20 = ndcg_metrics["ndcg@20"]


            print(f"\n  📈 Validation results:")

            main_metrics = ['@5', '@10', '@20']
            for metric in main_metrics:
                for key, value in logs.items():
                    if metric in key and ('NDCG' in key or 'Recall' in key):
                        print(f"    - {key}: {value:.4f}")
            
            is_best = False
            if ndcg_at_10 > best_ndcg_10:
                is_best = True
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)
                print(f"\n  🎯 New best model!")
                print(f"     NDCG@5: {ndcg_at_5:.4f} | NDCG@10: {best_ndcg_10:.4f} | NDCG@20: {ndcg_at_20:.4f}")
                print(f"     Model saved to: {best_state_path}")
            else:
                print(f"\n  Current - NDCG@5: {ndcg_at_5:.4f} | NDCG@10: {ndcg_at_10:.4f} | NDCG@20: {ndcg_at_20:.4f}")
                print(f"  Best - NDCG@10: {best_ndcg_10:.4f} (Epoch {best_epoch + 1})")
                print(f"  Patience: {epoch - best_epoch}/{trainer_config['patience']}")

            writer.log(logs)

            # save from time to time
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print(f"\n{'='*80}")
            print(f"⏸️  Early stopping: patience exhausted")
            print(f"  - Best epoch: {best_epoch + 1}")
 print(f" - NDCG@10: {best_ndcg_10:.4f}")
            print(f"{'='*80}\n")
            break

    print(f"\n{'='*80}")
    print(f"🧪 Evaluating best model on test set...")
    print(f"  - Best model from Epoch {best_epoch + 1}")
    print(f"  - Best on validation NDCG@10: {best_ndcg_10:.4f}")
    print(f"{'='*80}\n")

    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    model.load_state_dict(torch.load(best_state_path), strict=True)

    logs, _ = evaluate_helper(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )


    print(f"\n{'='*80}")
    print(f"📊 Final test evaluation results:")
    print(f"{'='*80}")
    for key, value in logs.items():
        if 'test' in key:
            print(f"  {key}: {value:.4f}")
    print(f"{'='*80}\n")

    writer.log(logs)
    writer.finish()

    return
