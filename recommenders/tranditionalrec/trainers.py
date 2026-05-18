# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import tqdm
import random
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        else:
            self.model.to(self.device)

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    # length: [length_lower_bound, length_upper_bound)
    def get_sample_scores_length(self, epoch, answers, pred_list, original_input_length, length_lower_bound, length_upper_bound):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        filter_pred_list = []
        for i in range(len(original_input_length)):  # length filter
            if length_lower_bound <= original_input_length[i] and original_input_length[i] < length_upper_bound:
                filter_pred_list.append(pred_list[i])
        pred_list = np.array(filter_pred_list)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        print(str(length_lower_bound) + " " + str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(length_lower_bound) + " " + str(post_fix) + '\n')
        return str(post_fix)

    def get_sample_scores(self, epoch, answers, pred_list, original_input_length):
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]
        for i in range(len(length_lower_bound)):
            self.get_sample_scores_length(epoch, answers, pred_list, original_input_length, length_lower_bound[i], length_upper_bound[i])
        # print(post_fix)
        # with open(self.args.log_file, 'a') as f:
        #     f.write(str(post_fix) + '\n')
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        # HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        # R_20 = recall_at_k(answers, pred_list, 20)
        # R_50 = recall_at_k(answers, pred_list, 50)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        return [R_5, R_10, R_20, NDCG_5, NDCG_10, NDCG_20, MRR_5, MRR_10, MRR_20], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(recall[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(recall[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(recall[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_topk_gpu(self, rating_pred, train_matrix, k=20):
        """GPU version of top-k ranking, avoiding CPU transfer"""
        # rating_pred: [batch_size, item_num]
        batch_size = rating_pred.size(0)

        if not isinstance(train_matrix, torch.Tensor):
            train_matrix = torch.tensor(train_matrix.toarray(), dtype=torch.float32, device=self.device)
        else:
            train_matrix = train_matrix.to(self.device)

        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        rating_pred = rating_pred * (1 - train_matrix[batch_indices.squeeze(1)])

        rating_pred = torch.where(train_matrix[batch_indices.squeeze(1)] > 0,
                                  torch.tensor(float('-inf'), device=self.device),
                                  rating_pred)

        topk_values, topk_indices = torch.topk(rating_pred, k, dim=1)

        return topk_indices

    def get_metric_gpu(self, pred_ranks, topk=10):
        """GPU version of metric calculation"""

        batch_size = pred_ranks.size(0)

        hit = (pred_ranks < topk).float().sum()

        mrr = torch.where(pred_ranks < topk,
                          1.0 / (pred_ranks + 1.0),
                          torch.tensor(0.0, device=self.device)).sum()

        ndcg = torch.where(pred_ranks < topk,
                          1.0 / torch.log2(pred_ranks + 2.0),
                          torch.tensor(0.0, device=self.device)).sum()

        return hit / batch_size, ndcg / batch_size, mrr / batch_size

    def get_sample_scores_gpu(self, epoch, pred_ranks, original_input_length):
        """GPU version of sample score calculation"""
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]

        all_results = []
        for i in range(len(length_lower_bound)):

            mask = (original_input_length >= length_lower_bound[i]) & \
                   (original_input_length < length_upper_bound[i])
            filtered_ranks = pred_ranks[mask]

            if len(filtered_ranks) > 0:
                R_5, NDCG_5, MRR_5 = self.get_metric_gpu(filtered_ranks, 5)
                R_10, NDCG_10, MRR_10 = self.get_metric_gpu(filtered_ranks, 10)
                R_20, NDCG_20, MRR_20 = self.get_metric_gpu(filtered_ranks, 20)

                post_fix = {
                    "Epoch": epoch,
                    "HR_5": '{:.7f}'.format(R_5.item()), "HR_10": '{:.7f}'.format(R_10.item()),
                    "HR_20": '{:.7f}'.format(R_20.item()),
                    "NDCG@5": '{:.7f}'.format(NDCG_5.item()), "NDCG@10": '{:.7f}'.format(NDCG_10.item()),
                    "NDCG@20": '{:.7f}'.format(NDCG_20.item()),
                    "MRR@5": '{:.7f}'.format(MRR_5.item()), "MRR@10": '{:.7f}'.format(MRR_10.item()),
                    "MRR@20": '{:.7f}'.format(MRR_20.item())
                }
                print(str(length_lower_bound[i]) + " " + str(post_fix))
                with open(self.args.log_file, 'a') as f:
                    f.write(str(length_lower_bound[i]) + " " + str(post_fix) + '\n')
                all_results.append(str(post_fix))

        R_5, NDCG_5, MRR_5 = self.get_metric_gpu(pred_ranks, 5)
        R_10, NDCG_10, MRR_10 = self.get_metric_gpu(pred_ranks, 10)
        R_20, NDCG_20, MRR_20 = self.get_metric_gpu(pred_ranks, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5.item()), "HR_10": '{:.7f}'.format(R_10.item()),
            "HR_20": '{:.7f}'.format(R_20.item()),
            "NDCG@5": '{:.7f}'.format(NDCG_5.item()), "NDCG@10": '{:.7f}'.format(NDCG_10.item()),
            "NDCG@20": '{:.7f}'.format(NDCG_20.item()),
            "MRR@5": '{:.7f}'.format(MRR_5.item()), "MRR@10": '{:.7f}'.format(MRR_10.item()),
            "MRR@20": '{:.7f}'.format(MRR_20.item())
        }
        return [R_5.item(), R_10.item(), R_20.item(), NDCG_5.item(), NDCG_10.item(), NDCG_20.item(),
                MRR_5.item(), MRR_10.item(), MRR_20.item()], str(post_fix)


class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}", colour="#00ff00")
        if train:
            self.model.train()
            avg_loss = 0.0
            rec_avg_loss = 0.0
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                if self.args.backbone == 'Narm':
                    sequence_output = self.model.finetune(input_ids)
                    test_item_emb = self.model.item_embeddings.weight
                    logits = torch.matmul(sequence_output, test_item_emb.transpose(0, 1))
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, target_pos[:, -1])
                else:
                    loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter))
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, _ = batch
                    recommend_output = self.model.finetune(input_ids)
                    if self.args.backbone != 'Narm':
                        recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    batch_pred_list = self.get_topk_gpu(rating_pred, self.args.train_matrix[user_ids.cpu().numpy()], k=20)

                    if i == 0:
                        pred_list = batch_pred_list.cpu().numpy()
                        answer_list = answers.cpu().numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list.cpu().numpy(), axis=0)
                        answer_list = np.append(answer_list, answers.cpu().numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                T = 0.0
                answer_list = None
                original_input_length = None
                pred_ranks_list = []
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, original_input_length_batch = batch
                    start_time = time.time()
                    recommend_output = self.model.finetune(input_ids)
                    end_time = time.time()
                    T += end_time - start_time
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)

                    batch_size = test_logits.size(0)

                    positive_logits = test_logits[:, 0:1]  # [batch_size, 1]
                    negative_logits = test_logits[:, 1:]   # [batch_size, 100]

                    ranks = torch.sum(positive_logits <= negative_logits, dim=1).float()

                    if i == 0:
                        pred_ranks_list = ranks
                        answer_list = answers
                        original_input_length = original_input_length_batch
                    else:
                        pred_ranks_list = torch.cat([pred_ranks_list, ranks], dim=0)
                        answer_list = torch.cat([answer_list, answers], dim=0)
                        original_input_length = torch.cat([original_input_length, original_input_length_batch], dim=0)

                return self.get_sample_scores_gpu(epoch, pred_ranks_list, original_input_length), T


def test_gpu_optimizations():
    """
    Helper function to test GPU optimizations
    Used to verify that GPU version of methods work correctly
    """
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return

    class MockModel:
        def __init__(self):
            self.item_embeddings = torch.nn.Embedding(100, 64).to(device)

    class MockArgs:
        def __init__(self):
            self.hidden_size = 64
            self.max_seq_length = 50
            self.train_matrix = None
            self.log_file = "/tmp/test_log.txt"

    mock_model = MockModel()
    mock_args = MockArgs()

    class TestTrainer:
        def __init__(self, model, args):
            self.model = model
            self.args = args
            self.device = device

        def get_topk_gpu(self, rating_pred, train_matrix, k=20):
            """GPU version of top-k ranking, avoiding CPU transfer"""
            # rating_pred: [batch_size, item_num]
            batch_size = rating_pred.size(0)

            if not isinstance(train_matrix, torch.Tensor):
                train_matrix = torch.tensor(train_matrix.toarray(), dtype=torch.float32, device=self.device)
            else:
                train_matrix = train_matrix.to(self.device)

            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
            rating_pred = rating_pred * (1 - train_matrix[batch_indices.squeeze(1)])

            rating_pred = torch.where(train_matrix[batch_indices.squeeze(1)] > 0,
                                      torch.tensor(float('-inf'), device=self.device),
                                      rating_pred)

            topk_values, topk_indices = torch.topk(rating_pred, k, dim=1)

            return topk_indices

        def get_metric_gpu(self, pred_ranks, topk=10):
            """GPU version of metric calculation"""

            batch_size = pred_ranks.size(0)

            hit = (pred_ranks < topk).float().sum()

            mrr = torch.where(pred_ranks < topk,
                              1.0 / (pred_ranks + 1.0),
                              torch.tensor(0.0, device=self.device)).sum()

            ndcg = torch.where(pred_ranks < topk,
                              1.0 / torch.log2(pred_ranks + 2.0),
                              torch.tensor(0.0, device=self.device)).sum()

            return hit / batch_size, ndcg / batch_size, mrr / batch_size

    test_trainer = TestTrainer(mock_model, mock_args)

    print("\n=== Testing get_metric_gpu ===")
    pred_ranks = torch.tensor([0, 1, 5, 15, 25], dtype=torch.float32, device=device)
    hit, ndcg, mrr = test_trainer.get_metric_gpu(pred_ranks, topk=10)
    print(f"Input ranks: {pred_ranks.cpu().numpy()}")
    print(f"HIT@10: {hit.item():.4f}")
    print(f"NDCG@10: {ndcg.item():.4f}")
    print(f"MRR@10: {mrr.item():.4f}")

    print("\n=== Testing get_topk_gpu ===")
    batch_size, item_num = 2, 10
    rating_pred = torch.randn(batch_size, item_num, device=device)

    from scipy.sparse import csr_matrix
    train_matrix = csr_matrix((batch_size, item_num))
    train_matrix[0, 0] = 1
    train_matrix[1, 1] = 1

    topk_indices = test_trainer.get_topk_gpu(rating_pred, train_matrix, k=5)
    print(f"Input shape: {rating_pred.shape}")
    print(f"Top-5 indices shape: {topk_indices.shape}")
    print(f"Top-5 indices: {topk_indices.cpu().numpy()}")

    print("\n=== GPU tests completed ===")
    print("All tests passed!")


if __name__ == "__main__":
    test_gpu_optimizations()
