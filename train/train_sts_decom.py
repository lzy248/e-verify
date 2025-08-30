import argparse
import logging
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys

sys.path.append(os.path.split(sys.path[0])[0])
from datetime import datetime
import math
from util import *
from config import device
from model.Embedding import DeEmbedder, DeSim, sim_calculator_dict, aggregator_dict
import torch
from train.train_sts_train_step import *
from train.train_sts_eval_step import eval_model
from train.train_sts_dataset import init_dataloader

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid tokenizer warning
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


set_random_seeds(seed_value=42)


def parse_args():
    parser = argparse.ArgumentParser()
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--name", type=str, default="", help="output file postfix")
    parser.add_argument(
        "--mode",
        type=str,
        default="triplet",
        choices=["triplet", "pair"],
        help="train mode",
    )
    # loss = alpha * tri_loss + (1 - alpha) * ce_loss
    # alpha = 1, only triplet loss
    # alpha = 0, only cross entropy loss
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="loss alpha between triplet and cross entropy",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        default="wiki_en",
        choices=["wiki_en"],
        help="dataset to train",
    )
    parser.add_argument(
        "--output_dir", type=str, default=f"experiment/output/", help="output dir"
    )
    parser.add_argument(
        "--n_keep", type=int, default=3, help="number of models to keep"
    )
    parser.add_argument(
        "--restore", type=str, default=None, help="path to restore model"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="transformer model for embedding",
    )
    parser.add_argument(
        "--max_len", type=int, default=512, help="max length of input tokens"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--aggregator",
        type=str,
        default="pma_ext",
        choices=aggregator_dict.keys(),
        help="aggregator for sentence embedding",
    )
    parser.add_argument(
        "--sim_calculator",
        type=str,
        default="iem_multi_v2",
        choices=sim_calculator_dict.keys(),
        help="calculator to calculate the similarity between two sentences",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="BCELoss",
        help="use BCEWithLogits when using cosine similarity otherwise BCELoss",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="hidden dimension of iem"
    )
    parser.add_argument(
        "--output_dim", type=int, default=1, help="output dimension of iem"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="number of heads in pma"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=2, help="output embeddings of pma"
    )
    parser.add_argument("--eval_every_percent", type=float, default=1)
    parser.add_argument("--eval_sim_threshold", type=float, default=0.5)
    parser.add_argument(
        "--ln", default=True, action="store_true", help="layer norm for pma"
    )
    parser.add_argument(
        "--norm", action="store_true", default=True, help="norm after sentence pooling"
    )
    parser.add_argument(
        "--margin", type=float, default=0.5, help="margin for triplet loss"
    )
    parser.add_argument("--drop_prob", type=float, default=0.1, help="dropout prob")
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="use lora for training bert embedding",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="rank for lora")

    parser.add_argument(
        "--lora_restore", type=str, default=None, help="the path to restore lora model"
    )

    parser.add_argument(
        "--eval_store_metric",
        type=str,
        choices=["acc", "loss", "f1", "auc", "none"],
        default="none",
        help="metric to store the best model",
    )
    parser.add_argument("--tensorboard", default=True, action="store_true")
    parser.add_argument("--tensorboard_dir", type=str, default=f"experiment/runs/")
    args = parser.parse_args()

    if args.name:
        args.output_dir = f"{args.output_dir}/{time}_{args.name}"
        args.tensorboard_dir = f"{args.tensorboard_dir}/{time}_{args.name}"
    else:
        args.output_dir = f"{args.output_dir}/{time}"
        args.tensorboard_dir = f"{args.tensorboard_dir}/{time}"

    if args.sim_calculator == "cosine":
        args.loss_fn = "BCEWithLogits"
    else:
        args.loss_fn = "BCELoss"
    return args


def train():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_args_to_json(args, os.path.join(args.output_dir, "args.json"))
    logger.info(f"Start training at {datetime.now()}")

    train_loader, val_loader = init_dataloader(args)

    # eval every certain ratio of train data, calc the steps
    eval_every_step = math.floor(len(train_loader) * args.eval_every_percent)

    logging.info(f"Eval every {eval_every_step} steps")

    embedder, sim_calculator, optimizer = init_models(args)

    best_metric = -1e9
    total_steps = 0

    if args.tensorboard:
        writer = SummaryWriter(comment="tensorboard", log_dir=args.tensorboard_dir)

    if args.restore:
        eval_model(
            args,
            val_loader,
            embedder,
            sim_calculator,
            best_metric,
            total_steps,
            save=False,
        )
    for current_epoch in trange(int(args.num_epochs), desc="Epoch", mininterval=0):
        torch.cuda.empty_cache()

        embedder.train()
        sim_calculator.train()

        batch_iterator = tqdm(
            train_loader,
            desc=f"Running Epoch {current_epoch + 1} of {args.num_epochs}",
            total=len(train_loader),
            mininterval=0,
        )
        for step, batch in enumerate(batch_iterator):
            optimizer.zero_grad()
            if args.mode == "triplet":
                loss, tri_loss, ce_loss = triplet_step(
                    args,
                    embedder,
                    sim_calculator,
                    batch,
                )
                batch_iterator.set_postfix(
                    {
                        "loss": "[{:4f}]".format(loss.item()),
                        "tri_loss": "[{:4f}]".format(tri_loss.item()),
                        "ce_loss": "[{:4f}]".format(ce_loss.item()),
                    }
                )
            elif args.mode == "pair":
                loss = pair_step(args, embedder, sim_calculator, batch)
                batch_iterator.set_postfix({"loss": "[{:4f}]".format(loss.item())})

            if args.tensorboard:
                writer.add_scalar("Train/Loss", loss, total_steps)

            loss.backward()
            optimizer.step()
            total_steps += 1
            if (step + 1) % eval_every_step == 0:
                metrics, best_metric = eval_model(
                    args,
                    val_loader,
                    embedder,
                    sim_calculator,
                    best_metric,
                    total_steps,
                )
                if args.tensorboard:
                    write_tensorboard(total_steps, writer, metrics)


def eval():
    args = parse_args()

    train_loader, val_loader = init_dataloader(args)
    embedder, sim_calculator, optimizer = init_models(args)

    best_metric = -1e9
    total_steps = 0
    eval_model(
        args,
        val_loader,
        embedder,
        sim_calculator,
        best_metric,
        total_steps,
        save=False,
    )


def write_tensorboard(total_steps, writer, metrics):
    acc, auc, f1, precision, recall, val_loss = metrics
    writer.add_scalar("Val/Accuracy", acc, total_steps)
    writer.add_scalar("Val/AUC", auc, total_steps)
    writer.add_scalar("Val/F1", f1, total_steps)
    writer.add_scalar("Val/Precision", precision, total_steps)
    writer.add_scalar("Val/Recall", recall, total_steps)
    writer.add_scalar("Val/Loss", val_loss, total_steps)


def init_models(args):
    if args.restore:
        embedder = DeEmbedder.load_from_ckpt(
            args.restore, add_lora=args.lora, lora_rank=args.lora_rank
        ).to(device)

        sim_calculator = DeSim.load_from_ckpt(args.restore).to(device)

    else:
        embedder = DeEmbedder.load_from_config(vars(args)).to(device)
        sim_calculator = DeSim.load_from_config(vars(args)).to(device)

    optimizer = torch.optim.Adam(
        list(embedder.parameters()) + list(sim_calculator.parameters()), lr=args.lr
    )

    return embedder, sim_calculator, optimizer


if __name__ == "__main__":
    train()
    # eval()
