from abc import ABC, abstractmethod
import os, sys

sys.path.append(os.path.split(sys.path[0])[0])

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Set the HF mirror endpoint

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np
from peft import PeftModel, LoraConfig, get_peft_model
from model.pro_model import *
from util import *
from config import device
from torch import nn


"""
The Embedding class is an abstract class that defines the interface for embedding models.
"""


class Embedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        pass


"""
The PairScore class is an abstract class that defines the interface for scoring pairs of texts.
"""


class PairScore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        pass


"""
Implementation of SentenceTransformerEmbedding class that inherits from Embedding class.
Using SentenceTransformer model to get embeddings of text.
"""


class SentenceTransformerEmbedding(Embedding):
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        self.model: SentenceTransformer = None
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            show_progress_bar=self.verbose,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )


class TransformerEmbedding(Embedding):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 32,
        verbose: bool = False,
        pooling_type: str = "mean",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name: str = model_name
        self.model: AutoModel = None
        self.tokenizer: AutoTokenizer = None
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        self.pooling_type: str = pooling_type
        self.normalize: bool = normalize
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device)
        self.model.eval()

    def _pooling_and_normalize(self, token_embeddings, mask):
        assert self.pooling_type in ["mean", "cls"], "Pooling type not supported."
        if self.pooling_type == "mean":
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
        elif self.pooling_type == "cls":
            sentence_embeddings = token_embeddings[:, 0]
        if self.normalize:
            # 归一化处理
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=-1
            )
        return sentence_embeddings

    @torch.no_grad()
    def get_embedding(self, texts: List[str]) -> np.ndarray:
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        embeddings = []
        for i in tqdm(
            range(num_batches),
            postfix=f"TransformerEmbedding:{self.model_name}",
            disable=not self.verbose,
        ):
            batch_texts = texts[i * self.batch_size : (i + 1) * self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            batch_embeddings = self._pooling_and_normalize(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embeddings.extend(batch_embeddings.cpu().numpy())
        return np.array(embeddings)


"""
Implementation of cosine_similarity function that calculates cosine similarity between two vectors.
"""


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    return np.dot(v1, v2)


"""
Implementation of batch_cosine_similarity function that calculates cosine similarity between two lists of vectors.
"""


def batch_cosine_similarity(
    embeddings1: List[List[float]], embeddings2: List[List[float]]
) -> List[float]:
    return np.diag(matrix_cosine_similarity(embeddings1, embeddings2))


def matrix_cosine_similarity(
    embeddings1: List[List[float]], embeddings2: List[List[float]], mode: str = "numpy"
) -> List[List[float]]:
    assert len(embeddings1) > 0, "Length of embeddings should be greater than 0."
    assert len(embeddings2) > 0, "Length of embeddings should be greater than 0."
    assert len(embeddings1[0]) == len(
        embeddings2[0]
    ), "Dimension of embeddings should be same."
    if mode == "numpy":
        return np.dot(embeddings1, np.array(embeddings2).T)
    elif mode == "torch":
        return (
            torch.mm(
                torch.tensor(embeddings1).to(device),
                torch.tensor(embeddings2).to(device).T,
            )
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose either 'numpy' or 'torch'.")


def filter_highest_score(scores: List[List[float]]) -> List[float]:
    """
    The filter_highest_score function takes in a list of scores and returns the highest score for each generation.

    Args:
    scores (List[List[float]]): A list of scores.

    Returns:
    List[float]: A list of the highest scores.
    """
    return [max(score) for score in scores]


def filter_highest_two_texts(
    premise_decomposed: List[str],
    hypothesis_decomposed: List[str],
    compare_fn: Embedding,
    verbose: bool = False,
    threshold: float = 0.6,
) -> List[Tuple[str, List[str]]]:
    """filter the highest similar score between the premise and hypothesis
       return tuple(hypothesis, [most similar premise, ...])
       to efficiently filter out unrelated premise

    Args:
        premise_decomposed (List[str]): the decomposed premise of reference texts
        hypothesis_decomposed (List[str]): the decomposed hypothesis of candidate texts
        compare_fn (Embedding): the embedding model to compare the texts
        verbose (bool, optional): show progress bar
        threshold (float, optional): the threshold to determine if the texts are similar.

    Returns:
        List[Tuple[str, List[str]]]: _description_
    """
    premise_embedding = compare_fn.get_embedding(premise_decomposed)
    hypothesis_embedding = compare_fn.get_embedding(hypothesis_decomposed)
    scores = matrix_cosine_similarity(hypothesis_embedding, premise_embedding)

    results = []

    for i, hypothesis in enumerate(hypothesis_decomposed):
        simlar_premise = []
        for j, score in enumerate(scores[i]):
            if score > threshold:
                simlar_premise.append(premise_decomposed[j])
        results.append((hypothesis, simlar_premise))

    return results


def compare_two_texts(
    premise_decomposed: List[str],
    hypothesis_decomposed: List[str],
    compare_fn: Union[Embedding, PairScore],
    integrate_highest_score: bool = True,
    verbose: bool = False,
):
    if isinstance(compare_fn, Embedding):
        premise_embedding = compare_fn.get_embedding(premise_decomposed)
        hypothesis_embedding = compare_fn.get_embedding(hypothesis_decomposed)
        scores = matrix_cosine_similarity(hypothesis_embedding, premise_embedding)
    elif isinstance(compare_fn, PairScore):
        scores = []
        for hypo in tqdm(
            hypothesis_decomposed, disable=not verbose, total=len(hypothesis_decomposed)
        ):
            hypo_list = [hypo] * len(premise_decomposed)
            score = compare_fn.get_score(list(zip(premise_decomposed, hypo_list)))
            scores.append(score)
    else:
        raise ValueError("compare_fn should be either Embedding or PairScore.")

    if integrate_highest_score:
        scores = filter_highest_score(scores)

    return scores


def compare_pairs(
    premise_decomposed: List[str],
    hypothesis_decomposed: List[str],
    compare_fn: Union[Embedding, PairScore],
):
    if isinstance(compare_fn, Embedding):
        premise_embedding = compare_fn.get_embedding(premise_decomposed)
        hypothesis_embedding = compare_fn.get_embedding(hypothesis_decomposed)
        scores = batch_cosine_similarity(premise_embedding, hypothesis_embedding)
    elif isinstance(compare_fn, PairScore):
        pairs = list(zip(premise_decomposed, hypothesis_decomposed))
        scores = compare_fn.get_score(pairs)
    else:
        raise ValueError("compare_fn should be either Embedding or PairScore.")

    return scores


aggregator_dict = {
    "pma": PMA,
    "pooling": Pooling,
    "pma_ext": PMAExt,
}
sim_calculator_dict = {
    "iem": IEM,
    "cosine": CosineSimilarity,
    "iem_d_p": D_P,
}


class DeEmbedder(Embedding, nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        aggregator,
        aggr_type: str = "pma",
        lora: bool = False,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        Embedding.__init__(self)
        nn.Module.__init__(self)
        self.model = model
        self.tokenizer = tokenizer
        self.aggregator: nn.Module = aggregator
        self.aggr_type = aggr_type
        self.lora: bool = lora
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        self._set_trainable()

    def _set_trainable(self):
        # Freeze all parameters in the embedding model
        for param in self.model.parameters():
            param.requires_grad = False

        if self.lora:
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        # Ensure that the parameters of the aggregator model are trainable
        for param in self.aggregator.parameters():
            param.requires_grad = True

    def forward(self, texts: List[str]) -> torch.Tensor:
        ebds, masks = bert_embedding(self.tokenizer, self.model, texts)
        aggr = self.aggregator(ebds, masks)
        return aggr

    def save_weight(self, path: str, steps: int):
        os.makedirs(path, exist_ok=True)
        if self.aggr_type.startswith("pma"):
            save_path = os.path.join(path, f"pma_checkpoint_step_{steps}.pth")
            torch.save(self.aggregator.state_dict(), save_path)
            logging.info(f"PMA model saved at {save_path}")

        if self.lora:
            save_path = os.path.join(path, f"lora_checkpoint_step_{steps}")
            self.model.save_pretrained(save_path)
            logging.info(f"LoRA model saved at {save_path}")

    @classmethod
    def load_from_config(cls, config: Dict):
        embedding_model = config.get("embedding_model", "BAAI/bge-base-en-v1.5")
        tokenizer = AutoTokenizer.from_pretrained(
            embedding_model, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True)
        logging.info(f"Embedding model loaded: {embedding_model}")
        if config.get("lora", False):
            lora_config = LoraConfig(
                r=config.get("lora_rank", 8),
                init_lora_weights="olora",
                target_modules=["query", "value"],
            )
            model = get_peft_model(model, lora_config)
            logging.info(f"Init LoRA with rank {config.get('lora_rank', 8)}")

        aggr_type = config.get("aggregator", "pma")
        aggregator = aggregator_dict[aggr_type].load_from_config(config)

        return cls(model, tokenizer, aggregator, aggr_type, config.get("lora", False))

    @classmethod
    def load_from_ckpt(
        cls,
        path: str,
        add_lora: bool = False,
        lora_rank: int = 8,
        n_latest: int = 1,
        with_cosine: bool = False,
    ):
        config_path = os.path.join(path, "args.json")
        config: Dict = read_json(config_path)
        embedding_model = config.get("embedding_model", "BAAI/bge-base-en-v1.5")
        tokenizer = AutoTokenizer.from_pretrained(
            embedding_model, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True)
        logging.info(f"Embedding model loaded: {embedding_model}")
        if config.get("lora", False):
            lora_path = get_latest_model(path, "lora", n_latest=n_latest)
            model = PeftModel.from_pretrained(model, lora_path)
            logging.info(f"LoRA model restored from {lora_path}")

        if add_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                init_lora_weights="olora",
                target_modules=["query", "value"],
                layers_to_transform=[10, 11],
            )
            model = get_peft_model(model, lora_config)
            logging.info(f"Init LoRA with rank {lora_rank}")

        aggr_type = config.get("aggregator", "pma")
        aggregator = aggregator_dict[aggr_type].load_from_ckpt(path, n_latest)

        return cls(model, tokenizer, aggregator, aggr_type, add_lora)

    @torch.no_grad()
    def get_embedding(
        self, texts: List[str], batch_size: int = 32, verbose: bool = False
    ) -> torch.Tensor:
        num_batches = (len(texts) + batch_size - 1) // batch_size
        embeddings = []

        for i in tqdm(range(num_batches), postfix="DeEmbedding", disable=not verbose):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            aggr = self.forward(batch_texts)
            embeddings.extend(aggr)

        embeddings = torch.stack(embeddings, dim=0)
        return embeddings


def bert_embedding(tokenizer, model, texts: List[str]):
    embeddings = []
    masks = []
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    embeddings.extend(outputs.last_hidden_state)
    masks.extend(inputs["attention_mask"])
    # 将list转换为tensor
    embeddings = torch.stack(embeddings, dim=0)
    masks = torch.stack(masks, dim=0)
    return embeddings, masks


class DeSim(torch.nn.Module):
    def __init__(
        self,
        sim_type: str,
        sim_model: nn.Module,
    ):
        super().__init__()
        self.sim_type = sim_type
        self.sim_model = sim_model

    def forward(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        return self.sim_model(embeddings1, embeddings2)

    def save_weight(self, path: str, steps: int):
        os.makedirs(path, exist_ok=True)
        if self.sim_type.startswith("iem"):
            save_path = os.path.join(path, f"iem_checkpoint_step_{steps}.pth")
            torch.save(self.sim_model.state_dict(), save_path)
            logging.info(f"IEM model saved at {save_path}")

    @torch.no_grad()
    def similar_score(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> torch.Tensor:
        num_batches = (len(embeddings1) + batch_size - 1) // batch_size
        scores = []
        for i in tqdm(range(num_batches), postfix="DeSim", disable=not verbose):
            batch_emb1 = embeddings1[i * batch_size : (i + 1) * batch_size]
            batch_emb2 = embeddings2[i * batch_size : (i + 1) * batch_size]
            score = self.sim_model(batch_emb1, batch_emb2)
            scores.extend(score)
        return torch.stack(scores, dim=0)

    @classmethod
    def load_from_config(cls, config: Dict):
        sim_type = config.get("sim_calculator", "iem")
        sim_model = sim_calculator_dict[sim_type].load_from_config(config)
        return cls(sim_model=sim_model, sim_type=sim_type)

    @classmethod
    def load_from_ckpt(cls, path: str, n_latest: int = 1):
        config_path = os.path.join(path, "args.json")
        config: Dict = read_json(config_path)

        sim_type = config.get("sim_calculator", "iem")
        sim_model = sim_calculator_dict[sim_type].load_from_ckpt(path, n_latest)
        return cls(sim_model=sim_model, sim_type=sim_type)


class DeBERTaScore(PairScore):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 32,
        verbose: bool = False,
        labels: List[str] = ["contradiction", "neutral", "entailment"],
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.verbose = verbose
        self.labels = labels
        self.entailment_index = self.labels.index("entailment")
        self.contradiction_index = self.labels.index("contradiction")
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(device)
        self.model.eval()

    """
    "0": "CONTRADICTION",
    "1": "NEUTRAL",
    "2": "ENTAILMENT"
    """

    @torch.no_grad()
    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        scores = []
        for i in tqdm(
            range(num_batches), postfix="DeBERTaScore", disable=not self.verbose
        ):
            batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]
            batch_pairs = [f"[CLS] {f} [SEP] {l} [SEP]" for f, l in batch_pairs]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_scores = self.model(**inputs, return_dict=True).logits
            batch_scores = (
                torch.softmax(
                    batch_scores[:, [self.entailment_index, self.contradiction_index]],
                    dim=1,
                )
                .to("cpu")
                .numpy()
                .astype(float)
            )
            scores.extend(batch_scores)
        out_scores = []
        for score in scores:
            out_scores.append(score[0])  # entailment
        return out_scores


class MiniCheckScore(PairScore):
    def __init__(
        self,
        model_name: str = "roberta-large",  # ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']
        batch_size: int = 32,
        max_model_len: int = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self._load_model()

    def _load_model(self):
        from minicheck.minicheck import MiniCheck

        self.checker = MiniCheck(
            model_name=self.model_name,
            batch_size=self.batch_size,
            max_model_len=self.max_model_len,
        )

    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ):
        docs = [d[0] for d in pairs]
        claims = [d[1] for d in pairs]
        _, scores, _, _ = self.checker.score(docs=docs, claims=claims)
        return scores


class FlanT5Score(PairScore):
    def __init__(
        self,
        model_name: str = "lytang/MiniCheck-Flan-T5-Large",
        batch_size: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.verbose = verbose
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        scores = []
        for i in tqdm(
            range(num_batches), postfix="FlanT5Score", disable=not self.verbose
        ):
            batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]
            batch_pairs = [
                f"predict: {self.tokenizer.eos_token.join([f, l])}"
                for f, l in batch_pairs
            ]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].size(0), 1), dtype=torch.long
            ).to(self.model.device)
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
            logits = outputs.logits.squeeze(1)
            label_logits = logits[:, torch.tensor([3, 209])].cpu()
            # 3 for no support and 209 for support
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            scores.extend(label_probs[:, 1].tolist())  # 1 for support
        return scores


class APIScore(PairScore):
    def __init__(
        self,
        api_name: str = "local",  # api name config in config.py
        verbose: bool = False,
        **kwargs,
    ):
        self.api_name = api_name
        self.verbose = verbose
        self.prompt = "Given the premise and the hypothesis. \nPremise: {premise}. \nHypothesis: {hypothesis}. \nPlease determine if the hypothesis is true based on the premise. \nAnswer(just answer 1 for true, 0 for false, no additional explaination): "
        # self.prompt = (
        #     "You are a helpful assistant. Your task is to determine whether the given hypothesis logically follows "
        #     "strictly from the provided premise.\n"
        #     "- Only use the information explicitly stated in the premise.\n"
        #     "- Do not rely on external knowledge, assumptions, or common sense.\n"
        #     "- If the hypothesis contains any subjective statements (e.g., emotions, opinions, evaluations), "
        #     "it must be directly and explicitly supported by the premise.\n"
        #     "- If the hypothesis cannot be entirely and directly inferred from the premise, consider it false.\n\n"
        #     "Premise: {premise}\n\n"
        #     "Hypothesis: {hypothesis}\n\n"
        #     "Answer (just answer 1 for true, 0 for false, no explanation):"
        # )
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        scores = []
        it = tqdm(pairs, postfix="APIScore", disable=not self.verbose)
        for pair in it:
            premise, hypothesis = pair
            prompt = self.prompt.format(premise=premise, hypothesis=hypothesis)
            answer, usage = query_gpt(prompt, api_name=self.api_name, return_usage=True)
            self.completion_tokens += usage["completion_tokens"]
            self.prompt_tokens += usage["prompt_tokens"]
            self.total_tokens += usage["total_tokens"]
            it.set_postfix(
                {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                }
            )
            if "1" in answer:
                scores.append(1.0)
            elif "0" in answer:
                scores.append(0.0)
            else:
                scores.append(0.0)

        return scores


class LLMScore(PairScore):
    def __init__(
        self,
        model_name: str = "",  # api name config in config.py
        vllm_boost: bool = False,
        verbose: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        self.model_name = model_name
        self.vllm_boost = vllm_boost
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.prompt = "Given the premise and the hypothesis. \nPremise: {premise}. \nHypothesis: {hypothesis}. \nPlease determine if the hypothesis is true based on the premise. \nAnswer(just answer 1 for true, 0 for false, no additional explaination): "
        # self.prompt = (
        #     "You are a helpful assistant. Your task is to determine whether the given hypothesis logically follows "
        #     "strictly from the provided premise.\n"
        #     "- Only use the information explicitly stated in the premise.\n"
        #     "- Do not rely on external knowledge, assumptions, or common sense.\n"
        #     "- If the hypothesis contains any subjective statements (e.g., emotions, opinions, evaluations), "
        #     "it must be directly and explicitly supported by the premise.\n"
        #     "- If the hypothesis cannot be entirely and directly inferred from the premise, consider it false.\n\n"
        #     "Premise: {premise}\n\n"
        #     "Hypothesis: {hypothesis}\n\n"
        #     "Answer (just answer 1 for true, 0 for false, no explanation):"
        # )
        self._init_model()

    def _init_model(self):
        logging.info(f"Loading model from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote=True,
            padding_side="left",
            max_length=512,
            # trucation=True,
            # padding=True,
        )
        if self.vllm_boost:
            from vllm import LLM

            self.model = LLM(model=self.model_name)
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto", device_map="auto"
                )
                .to(device)
                .eval()
            )

    def _format_input(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        scores = []

        inputs = [
            self._format_input(self.prompt.format(premise=p[0], hypothesis=p[1]))
            for p in pairs
        ]
        if self.vllm_boost:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=8,
            )
            generated_ids = self.model.generate(
                inputs,
                sampling_params=sampling_params,
            )
            scores = [
                1.0 if "1" in item.outputs[0].text else 0.0 for item in generated_ids
            ]
        else:
            for i in tqdm(
                range(0, len(inputs), self.batch_size),
                postfix=self.model_name,
                disable=not self.verbose,
            ):
                batch_texts = inputs[i : i + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    max_length=4096,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=8)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                scores = [1.0 if "1" in item else 0.0 for item in response]

        return scores


class DePairScore(PairScore):
    def __init__(
        self,
        ebedder: DeEmbedder,
        sim_calculator: DeSim,
        batch_size: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        self.embedder = ebedder.to(device).eval()
        self.sim_calculator = sim_calculator.to(device).eval()

    @classmethod
    def load_from_ckpt(
        cls, path: str, verbose: bool = False, batch_size: int = 32, n_latest: int = 1
    ):
        embedder = DeEmbedder.load_from_ckpt(path, n_latest=n_latest)
        sim_calculator = DeSim.load_from_ckpt(path, n_latest=n_latest)
        return cls(embedder, sim_calculator, batch_size, verbose)

    @torch.no_grad()
    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        scores = []
        for i in tqdm(
            range(num_batches), postfix="DePairScore", disable=not self.verbose
        ):
            batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]
            len_pair = len(batch_pairs)

            former = [pair[0] for pair in batch_pairs]
            latter = [pair[1] for pair in batch_pairs]
            conbined_text = former + latter

            combined_emb = self.embedder.get_embedding(conbined_text)

            former_emb = combined_emb[:len_pair]
            latter_emb = combined_emb[len_pair:]

            score = self.sim_calculator(former_emb, latter_emb)
            scores.extend(score.cpu().numpy())
        return scores


class BertScore(PairScore):
    def __init__(
        self,
        batch_size: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size: int = batch_size
        self.verbose: bool = verbose

    @torch.no_grad()
    def get_score(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        from bert_score import score

        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        scores = []
        for i in tqdm(
            range(num_batches), postfix="BertScore", disable=not self.verbose
        ):
            batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]
            refs = [pair[0] for pair in batch_pairs]
            hyps = [pair[1] for pair in batch_pairs]
            _, _, f1 = score(hyps, refs, lang="en", verbose=False)
            scores.extend(f1.cpu().numpy())
        return scores


if __name__ == "__main__":

    model = APIScore(api_name="")
