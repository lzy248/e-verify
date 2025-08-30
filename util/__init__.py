import json
import random
import logging
import os

import sys

sys.path.append(os.path.split(sys.path[0])[0])
from typing import List, Dict, Any, Union, Optional, Tuple
import shutil
import openai

from config import api_list, device
import torch
import numpy as np
from transformers import AutoConfig
from tqdm import tqdm
from retry import retry

logging.basicConfig(level=logging.INFO)


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_random_seeds(42)


def save_args_to_json(args, filename):
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=4)


def read_jsonl(path: str, sample: int = None, verbose: bool = False) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding line: {line.strip()} - {e}")

    if sample and len(data) > sample:
        data = random.sample(data, sample)
    if verbose:
        logging.info(f"数据集: {path}")
    if sample and verbose:
        logging.info(f"采样数据集大小: {len(data)}")
    elif verbose:
        logging.info(f"数据集大小: {len(data)}")
    return data


def write_jsonl(data: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: List[Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_by_type(path: str):
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    elif path.endswith(".json"):
        return read_json(path)


# to suppress the httpx logger
httpx_logger: logging.Logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


@retry(tries=3, delay=1)
def query_gpt(
    prompt: str,
    temperature: float = 0.7,
    api_name: str = "local",
    return_usage: bool = False,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    model = api_list[api_name]["OPENAI_MODEL"]
    api_key = api_list[api_name]["OPENAI_API_KEY"]
    base_url = api_list[api_name]["OPENAI_BASE_URL"]
    openai.api_key = api_key
    openai.base_url = base_url

    chat_completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    if return_usage:
        usage = {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
        }
        return chat_completion.choices[0].message.content, usage
    else:
        return chat_completion.choices[0].message.content


def get_latest_model(save_dir, prefix="model", n_latest=1) -> Union[str, List[str]]:
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith(f"{prefix}_checkpoint_step_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if len(checkpoint_files) >= n_latest:
        latest_model = checkpoint_files[-n_latest]
        model_path = os.path.join(save_dir, latest_model)
        return model_path
    else:
        return None


def triplet_loss(neg_score, pos_score, margin):
    return torch.nn.functional.relu(margin + neg_score - pos_score).mean()


BCELoss = None
BCELossWithLogits = None


def cross_entropy_loss(scores, labels, loss_fn="BCELoss"):
    global BCELoss, BCELossWithLogits
    scores = scores.float()
    labels = labels.float()
    if loss_fn == "BCELoss":
        if BCELoss is None:
            BCELoss = torch.nn.BCELoss()
        return BCELoss(scores, labels)
    elif loss_fn == "BCEWithLogits":
        if BCELossWithLogits is None:
            BCELossWithLogits = torch.nn.BCEWithLogitsLoss()
        return BCELossWithLogits(scores, labels)
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")


def calc_loss(args, neg_score, pos_score):
    tri_loss = triplet_loss(
        neg_score=neg_score, pos_score=pos_score, margin=args.margin
    )

    scores = torch.cat((pos_score, neg_score), dim=0)
    labels = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score)), dim=0)

    ce_loss = cross_entropy_loss(scores, labels, args.loss_fn)

    loss = args.loss_alpha * tri_loss + (1 - args.loss_alpha) * ce_loss
    return loss, tri_loss, ce_loss


def delete_old_models(save_dir, n_keep=1, prefix="model"):
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith(f"{prefix}_checkpoint_step_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if len(checkpoint_files) > n_keep:
        files_to_remove = checkpoint_files[: len(checkpoint_files) - n_keep]
        for file_name in files_to_remove:
            total_file_path = os.path.join(save_dir, file_name)
            if os.path.isdir(total_file_path):
                shutil.rmtree(total_file_path)
            else:
                os.remove(total_file_path)


def get_model_emb_dim(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "hidden_size"):
        embedding_dim = config.hidden_size
    elif hasattr(config, "d_model"):
        embedding_dim = config.d_model
    else:
        raise ValueError(f"Can not get embedding dimension from model: {model_name}.")
    return embedding_dim


# to suppress the stanza logger
logging.getLogger("stanza").setLevel(logging.ERROR)
import stanza

# to suppress the stanza's FutureWarning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

nlp = None


def stanza_split_sentence(text: str) -> List[str]:
    """split text into sentences using stanza

    Args:
        text (str): input text

    Returns:
        List[str]: list of sentences
    """
    global nlp
    if nlp is None:
        nlp = stanza.Pipeline(lang="en", processors="tokenize", download_method=None)
    doc = nlp(text)
    return [sent.text for sent in doc.sentences]


def split_sentence_by_len(
    text: str, max_len: int = 512, overlap: int = 0, sep=None
) -> List[str]:
    """split text into sentences by length

    Args:
        text (str): input text
        max_len (int, optional): max length of sentence. Defaults to 512.

    Returns:
        List[str]: list of sentences
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=sep,
        chunk_size=max_len,
        chunk_overlap=overlap,
        keep_separator="end",
    )
    return splitter.split_text(text)


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def eval(model, y_pred, y_true, print_result=True):
    metrics = {
        "acc": accuracy_score(y_true=y_true, y_pred=y_pred),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average="macro"),
        "recall": recall_score(
            y_true=y_true, y_pred=y_pred, zero_division=0, average="macro"
        ),
        "precision": precision_score(
            y_true=y_true, y_pred=y_pred, zero_division=0, average="macro"
        ),
    }

    if print_result:
        print(f"model: {model:<40}", end=" ")
        for metric_name, metric_value in metrics.items():
            end = "\n" if metric_name == "precision" else " "
            print(f"{metric_name}: {metric_value:.4f}", end=end)

    return {"model": model, **metrics}


if __name__ == "__main__":
    print(
        query_gpt(
            "What is the capital of France?",
            temperature=0.7,
            api_name="gpt-4o",
            return_usage=True,
        )
    )
    pass
