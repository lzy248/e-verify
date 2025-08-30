import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Dict, Any, Union
import os
from util import read_json, get_model_emb_dim, get_latest_model
import logging
from config import device, pooling_type_dict


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

    def forward(self, Q, K, pad_mask=None):
        Q_ = self.fc_q(Q)
        K_, V_ = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)
        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
        score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = Q + O
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class PMA(nn.Module):

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_seeds: int = 1,
        ln: bool = True,
        norm: bool = True,
    ):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.norm = norm

    # X: (batch_size, sentence_len, embedding_dim)
    # pad_mask: (batch_size, sentence_len)
    def forward(self, X, pad_mask):
        if self.S.dtype != torch.bfloat16:
            X = X.float()
        emb = self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)
        emb = emb.squeeze(1)
        if self.norm:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    @classmethod
    def load_from_state_dict(cls, state_dict_path: str, config: Dict):
        model = cls.load_from_config(config)
        state_dict = torch.load(state_dict_path, weights_only=True)
        model.load_state_dict(state_dict)
        logging.info(f"PMA model restored from {state_dict_path}")
        return model

    @classmethod
    def load_from_ckpt(cls, file_path: str, n_latest: int = 1):
        config_path = os.path.join(file_path, "args.json")
        config = read_json(config_path)
        model_path = get_latest_model(file_path, prefix="pma", n_latest=n_latest)
        model = cls.load_from_state_dict(model_path, config)
        return model

    @classmethod
    def load_from_config(cls, config: Dict):
        dim = get_model_emb_dim(config.get("embedding_model", "BAAI/bge-base-en-v1.5"))
        logging.info(
            f"Init PMA model with dim={dim}, num_heads={config.get('num_heads',8)}, num_seeds={config.get('num_seeds',1)}, ln={config.get('ln',True)}, norm={config.get('norm',True)}"
        )
        return cls(
            dim=dim,
            num_heads=config.get("num_heads", 8),
            num_seeds=config.get("num_seeds", 1),
            ln=config.get("ln", True),
            norm=config.get("norm", True),
        )


class PMAExt(PMA):

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_seeds: int = 2,
        ln: bool = True,
        norm: bool = True,
        pooling_type: str = "cls",
    ):
        super(PMAExt, self).__init__(dim, num_heads, num_seeds, ln, norm)
        self.pooling_type = pooling_type

    def forward(self, X, pad_mask):
        if self.S.dtype != torch.bfloat16:
            X = X.float()
        emb = self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)
        # emb = emb.squeeze(1)
        if self.norm:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        # cls or mean pooling
        if self.pooling_type == "mean":
            pooling = self.mean_pooling(X, pad_mask)
        elif self.pooling_type == "cls":
            pooling = self.cls_pooling(X, pad_mask)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")
        pooling = self.normalize(pooling)

        # 放到 emb[bs,num_seeds,dim]中, [bs,num_seeds,dim] + [bs,dim] -> [bs,num_seeds+1,dim]
        emb = torch.cat((emb, pooling.unsqueeze(1)), dim=1)
        return emb

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def normalize(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def cls_pooling(self, token_embeddings, mask):
        return token_embeddings[:, 0]

    @classmethod
    def load_from_config(cls, config: Dict):
        embedding_model = config.get("embedding_model", "BAAI/bge-base-en-v1.5")
        dim = get_model_emb_dim(embedding_model)
        pooling_type = pooling_type_dict.get(embedding_model, "mean")
        logging.info(
            f"Init PMA model with dim={dim}, num_heads={config.get('num_heads',8)}, num_seeds={config.get('num_seeds',2)}, ln={config.get('ln',True)}, norm={config.get('norm',True)}, pooling_type={pooling_type}"
        )
        return cls(
            dim=dim,
            num_heads=config.get("num_heads", 8),
            num_seeds=config.get("num_seeds", 2),
            ln=config.get("ln", True),
            norm=config.get("norm", True),
            pooling_type=pooling_type,
        )


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        logging.info("Pooling model initialized")

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def cls_pooling(self, token_embeddings, mask):
        return token_embeddings[:, 0]

    def normalize(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def forward(self, X, mask):
        emb = self.cls_pooling(X, mask)
        emb = self.normalize(emb)
        return emb

    @classmethod
    def load_from_config(cls, *_):
        return cls()

    @classmethod
    def load_from_ckpt(cls, *_):
        return cls()


class IEM(nn.Module):

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 512,
        d_output: int = 1,
        drop_prob: float = 0.0,
    ):
        super(IEM, self).__init__()
        self.linear1 = nn.Linear(2 * dim, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, d_output)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, premise, hypothesis):
        x = torch.cat((premise, hypothesis), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.relu(x)
        logits = self.linear2(x)
        score = self.sigmoid(logits)
        score = score.squeeze(-1)  # 去除最后一个维度
        return score

    @classmethod
    def load_from_state_dict(cls, state_dict_path: str, config: Dict):
        model = cls.load_from_config(config)
        state_dict = torch.load(state_dict_path, weights_only=True)
        model.load_state_dict(state_dict)
        logging.info(f"IEM model restored from {state_dict_path}")
        return model

    @classmethod
    def load_from_ckpt(cls, file_path: str, n_latest: int = 1):
        config_path = os.path.join(file_path, "args.json")
        config = read_json(config_path)
        dim = get_model_emb_dim(config["embedding_model"])
        model_path = get_latest_model(file_path, prefix="iem", n_latest=n_latest)
        model = cls.load_from_state_dict(model_path, config)
        return model

    @classmethod
    def load_from_config(cls, config: Dict):
        dim = get_model_emb_dim(config.get("embedding_model", "BAAI/bge-base-en-v1.5"))
        logging.info(
            f"Init IEM model with dim={dim}, hidden={config.get('hidden_dim',512)}, d_output={config.get('output_dim',1)}, drop_prob={config.get('drop_prob',0)}"
        )
        return cls(
            dim=dim,
            hidden=config.get("hidden_dim", 512),
            d_output=config.get("output_dim", 1),
            drop_prob=config.get("drop_prob", 0.1),
        )


class D_P(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        hidden: int = 512,
        d_output: int = 1,
        drop_prob: float = 0.0,
    ):
        super(D_P, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * 128, d_output),
            nn.Sigmoid(),
        )

    def forward(self, premise, hypothesis):
        if premise.ndim == 2 and hypothesis.ndim == 2:  # compat with pooling
            # 如果是 (bs, dim)，则扩展为 (bs, 2, dim)
            premise = premise.unsqueeze(1).expand(-1, 2, -1)
            hypothesis = hypothesis.unsqueeze(1).expand(-1, 2, -1)
        pre1 = premise[:, 0, :]
        pre2 = premise[:, 1, :]

        hyp1 = hypothesis[:, 0, :]
        hyp2 = hypothesis[:, 1, :]

        # 1: emb1 - emb2
        pair = torch.cat([pre1, hyp1], dim=-1)  # (bs, dim*2)
        result1 = self.seq1(pair)

        # 2: emb1 - emb2
        diff = pre2 - hyp2  # (bs, dim)
        result2 = self.seq2(diff)

        # out
        stacked_results = torch.cat([result1, result2], dim=-1)  # (bs, 2*d_output)
        # Pass through the final output layer
        final_output = self.output(stacked_results).squeeze(-1)  # (bs)

        return final_output  # (bs,)


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        logging.info("CosineSimilarity model initialized")

    def forward(self, premise, hypothesis):
        if (
            premise.ndim == 3
            and hypothesis.ndim == 3
            and premise.shape[1] == 3
            and hypothesis.shape[1] == 3
        ):
            # 如果是 (bs, 3, dim)，则将三个维度拼接起来变成 (bs, 3*dim)
            premise = premise.view(premise.shape[0], -1)
            hypothesis = hypothesis.view(hypothesis.shape[0], -1)

        premise = torch.nn.functional.normalize(premise, p=2.0, dim=-1)
        hypothesis = torch.nn.functional.normalize(
            hypothesis, p=2.0, dim=-1
        )  # normalize

        cosine_sim = torch.diag(torch.mm(premise, hypothesis.T))  # (bs, )
        return cosine_sim  # not normalized to [0,1], use bce_with_logits_loss to calculate loss

    @classmethod
    def load_from_config(cls, *_):
        return cls()

    @classmethod
    def load_from_ckpt(cls, *_):
        return cls()
