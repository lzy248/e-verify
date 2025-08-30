import torch
from torch.utils.data import DataLoader, Dataset
import os
from util import delete_old_models
import logging
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from config import device
import numpy as np
from util import cross_entropy_loss, triplet_loss, calc_loss


@torch.no_grad()
def eval_model(
    args,
    val_loader: DataLoader,
    embedder,
    sim_calculator,
    best_metric,
    total_steps: int,
    save=True,
):
    # default to pair evaluation
    acc, auc, f1, precision, recall, val_loss = eval_model_pair(
        args,
        embedder,
        sim_calculator,
        val_loader,
        threshold=args.eval_sim_threshold,
    )
    logging.info(
        f"Val Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Loss: {val_loss:.4f}"
    )
    save_flag = False
    if args.eval_store_metric == "f1":
        current_metric = f1
    elif args.eval_store_metric == "acc":
        current_metric = acc
    elif args.eval_store_metric == "loss":
        current_metric = (
            -val_loss
        )  # to match the following metric comparison, use negative value
    elif args.eval_store_metric == "auc":
        current_metric = auc
    elif args.eval_store_metric == "none":
        save_flag = True
        current_metric = -val_loss
    if (current_metric >= best_metric or save_flag) and save:
        best_metric = current_metric
        sim_calculator.save_weight(args.output_dir, total_steps)
        embedder.save_weight(args.output_dir, total_steps)
        delete_old_models(args.output_dir, n_keep=args.n_keep, prefix="pma")
        delete_old_models(args.output_dir, n_keep=args.n_keep, prefix="iem")
        delete_old_models(args.output_dir, n_keep=args.n_keep, prefix="lora")

    return (acc, auc, f1, precision, recall, val_loss), best_metric


def eval_model_pair(args, embedder, sim_calculator, val_loader, threshold=0.5):
    embedder.eval()
    sim_calculator.eval()
    all_labels = []
    all_preds = []

    loss = []

    for batch in tqdm(val_loader):
        premise, hypothesis, labels = batch

        all_labels.extend(labels)

        combined_text = premise + hypothesis
        combined_emb = embedder(combined_text)

        premise_emb = combined_emb[: len(premise)]
        hypothesis_emb = combined_emb[len(premise) :]

        pair_scores = sim_calculator(premise_emb, hypothesis_emb)

        all_preds.extend(pair_scores)

        labels = labels.to(device)
        # logging.info(f"pair_scores: {pair_scores}")
        # logging.info(f"scores: {scores}")
        loss.append(cross_entropy_loss(pair_scores, labels, args.loss_fn))

    all_preds = [score.item() for score in all_preds]
    all_labels = [score.item() for score in all_labels]

    binary_preds = [1 if score > threshold else 0 for score in all_preds]
    binary_labels = [1 if score > threshold else 0 for score in all_labels]

    acc = accuracy_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall = recall_score(binary_labels, binary_preds, zero_division=0)

    if len(set(binary_labels)) == 1:
        auc = 0
    else:
        auc = roc_auc_score(binary_labels, all_preds)
    loss = [l.item() for l in loss]
    embedder.train()
    sim_calculator.train()
    return acc, auc, f1, precision, recall, np.mean(loss)
