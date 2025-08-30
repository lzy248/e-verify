from util import calc_loss, cross_entropy_loss
from config import device
import torch
import numpy as np


def triplet_step(args, embedder, sim_calculator, batch):
    anchor, pos, neg = batch

    combined_text = anchor + pos + neg

    combined_emb = embedder(combined_text)

    anchor_pma = combined_emb[: len(anchor)]
    pos_pma = combined_emb[len(anchor) : len(anchor) + len(pos)]
    neg_pma = combined_emb[len(anchor) + len(pos) :]

    pos_score = sim_calculator(anchor_pma, pos_pma)
    neg_score = sim_calculator(anchor_pma, neg_pma)

    loss, tri_loss, ce_loss = calc_loss(args, neg_score, pos_score)

    return loss, tri_loss, ce_loss


def pair_step(args, embedder, sim_calculator, batch):
    premise, hypothesis, scores = batch
    combined_text = premise + hypothesis
    combined_emb = embedder(combined_text)

    premise_pma = combined_emb[: len(premise)]
    hypothesis_pma = combined_emb[len(premise) :]

    pair_scores = sim_calculator(premise_pma, hypothesis_pma)

    scores = scores.to(device)

    loss = cross_entropy_loss(pair_scores, scores)

    return loss
