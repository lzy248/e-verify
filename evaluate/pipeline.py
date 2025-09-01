import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用的GPU编号
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.path.split(sys.path[0])[0])

from model.Embedding import (
    Embedding,
    PairScore,
    TransformerEmbedding,
    DeBERTaScore,
    DePairScore,
    matrix_cosine_similarity,
)
from util import read_jsonl, write_jsonl, read_json, write_json

from model.Decomposer import Decomposer, T5Decomposer
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)


from model.Embedding import *


def judge_support_or_refute(scores: List[float], threshold: float = 0.5) -> str:
    """
    The judge_support_or_refute function takes in a list of scores and returns a label indicating whether the premise supports the hypothesis.

    Args:
    scores (List[float]): A list of scores.
    threshold (float): A float value representing the threshold.

    Returns:
    out (SUPPORTS or REFUTES): A label indicating whether the premise supports the hypothesis.
    """
    if any(score < threshold for score in scores):
        return "contradiction"
    else:
        return "entailment"


def judge_wiki_en(
    judge_model,
    input_path="data/wiki_en_sentence/preprocessed/test_data_pair.json",
    output_path="data/wiki_en_sentence/eval",
):
    data = read_json(input_path)
    os.makedirs(output_path, exist_ok=True)

    for name, fun in judge_model.items():
        results = []
        print(f"Running {name}")
        model_class = fun["model_class"]
        model_args = fun["model_args"]
        model = model_class(**model_args)

        start = time.time()
        premise_list = [d[0] for d in data]
        hypothesis_list = [d[1] for d in data]
        label_list = ["entailment" if d[2] == 1 else "contradiction" for d in data]

        score_list = compare_pairs(premise_list, hypothesis_list, model)

        for s, l in zip(score_list, label_list):
            item = {}
            item["label"] = l
            item["score"] = [float(s)]
            results.append(item)

        total_time = time.time() - start
        print("Total time: ", total_time)
        write_json(results, f"{output_path}/{name}.json")
        torch.cuda.empty_cache()


def judge_wiki_hallu(
    judge_model,
    input_path="data/wiki_bio_hallu/wiki_bio_simple.json",
    output_base_path="data/wiki_bio_hallu/eval_simple",
):
    data = read_json(input_path)
    type_dict = {
        "claim": "wiki_decomposed",
        "split": "splitted_wiki",
        "doc": "wiki_bio_text",
        "splitted_wiki": "splitted_wiki",
        "wiki_decomposed": "wiki_decomposed",
    }
    for name, fun in judge_model.items():
        results = []
        print(f"Running {name}")
        model_class = fun["model_class"]
        model_args = fun["model_args"]
        model = model_class(**model_args)
        premise_key = type_dict[fun.get("premise_key", "wiki_decomposed")]
        start = time.time()
        it = tqdm(data, total=len(data))
        for d in it:
            item = {}
            # item["idx"] = d["idx"]
            item["labels"] = d["labels"]
            premise = d[premise_key]
            hypothesis = d["text_with_error_decomposed"]
            if isinstance(premise, str):
                premise = [premise]
            if isinstance(hypothesis, str):
                hypothesis = [hypothesis]
            p_score = compare_two_texts(
                premise_decomposed=premise,
                hypothesis_decomposed=hypothesis,
                compare_fn=model,
            )
            p_score = [float(score) for score in p_score]
            item["score"] = p_score
            results.append(item)
            if isinstance(model, APIScore):
                it.set_postfix(
                    postfix={
                        "chat_completion": model.completion_tokens,
                        "prompt_tokens": model.prompt_tokens,
                        "total_tokens": model.total_tokens,
                    }
                )
        total_time = time.time() - start
        print("Total time: ", total_time)
        write_json(results, f"{output_base_path}/{name}.json")
        torch.cuda.empty_cache()


def judge_llm_aggrefact(
    judge_model,
    input_path="data/llm_aggrefact/preprocessed/aggrefact_cnn.json",
    output_path="data/llm_aggrefact/eval_aggrefact_cnn",
):
    data = read_json(input_path)
    os.makedirs(output_path, exist_ok=True)
    type_dict = {
        "claim": "doc_decomposed",
        "split": "doc_split",
        "doc": "doc",
        "doc_split": "doc_split",
        "doc_decomposed": "doc_decomposed",
    }
    for name, fun in judge_model.items():
        results = []
        print(f"Running {name}")
        model_class = fun["model_class"]
        model_args = fun["model_args"]
        model = model_class(**model_args)
        premise_key = type_dict[fun.get("premise_key", "doc_decomposed")]
        start = time.time()
        it = tqdm(data, total=len(data))
        for d in it:
            item = {}
            item["label"] = d["label"]
            premise = d[premise_key]
            hypothesis = d["claim_decomposed"]
            if isinstance(premise, str):
                premise = [premise]
            if isinstance(hypothesis, str):
                hypothesis = [hypothesis]
            p_score = compare_two_texts(
                premise_decomposed=premise,
                hypothesis_decomposed=hypothesis,
                compare_fn=model,
            )
            p_score = [float(score) for score in p_score]
            item["score"] = p_score
            results.append(item)
            if isinstance(model, APIScore):
                it.set_postfix(
                    postfix={
                        "chat_completion": model.completion_tokens,
                        "prompt_tokens": model.prompt_tokens,
                        "total_tokens": model.total_tokens,
                    }
                )
        total_time = time.time() - start
        print("Total time: ", total_time)
        write_json(results, f"{output_path}/{name}.json")
        torch.cuda.empty_cache()


if __name__ == "__main__":

    judge_model = {
        # "deberta_mnli_fever_anli_split": {
        #     "model_class": DeBERTaScore,
        #     "model_args": {
        #         "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        #         "labels": ["entailment", "neutral", "contradiction"],
        #     },
        #     "premise_key": "doc_split",  # "splitted_wiki",
        # },
        # "nli-deberta-v3-base_split": {
        #     "model_class": DeBERTaScore,
        #     "model_args": {
        #         "model_name": "cross-encoder/nli-deberta-v3-base",
        #         "labels": ["contradiction", "entailment", "neutral"],
        #     },
        #     "premise_key": "doc_split",  # "splitted_wiki",
        # },
        # "bge_emb": {
        #     "model_class": TransformerEmbedding,
        #     "model_args": {
        #         "model_name": "BAAI/bge-base-en-v1.5",
        #         "pooling_type": "cls",
        #     },
        # },
        # "bert_score": {"model_class": BertScore, "model_args": {}},
        # "qwen7b": {
        #     "model_class": APIScore,
        #     "model_args": {"api_name": "local", "verbose": True},
        #     "premise_key": "doc",  # "wiki_bio_text",
        # },
        # "qwen2.5-0.5b": {
        #     "model_class": LLMScore,
        #     "model_args": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "verbose": True},
        #     "premise_key": "doc",
        # },
        # "qwen2.5-7b": {
        #     "model_class": LLMScore,
        #     "model_args": {"model_name": "Qwen/Qwen2.5-7B-Instruct", "verbose": True},
        #     "premise_key": "doc",
        # },
        # "gpt-4o": {
        #     "model_class": APIScore,
        #     "model_args": {"api_name": "gpt-4o", "verbose": True},
        #     "premise_key": "doc",  # "wiki_bio_text",
        # },
        "d_p_lora": {
            "model_class": DePairScore.load_from_ckpt,
            "model_args": {
                "path": "experiment/ckpt/d_p_lora",
            },
        },
        # "minicheck_deberta": {
        #     "model_class": MiniCheckScore,
        #     "model_args": {
        #         "model_name": "deberta-v3-large",
        #     },
        #     "premise_key": "split",
        # },
        # "minicheck_flan-t5": {
        #     "model_class": MiniCheckScore,
        #     "model_args": {
        #         "model_name": "flan-t5-large",
        #     },
        #     "premise_key": "doc",
        # },
        # "roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli": {
        #     "model_class": DeBERTaScore,
        #     "model_args": {
        #         "model_name": "garak-llm/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        #         "labels": ["entailment", "neutral", "contradiction"],
        #     },
        #     "premise_key": "split",
        # },
        # "nli-roberta-base": {
        #     "model_class": DeBERTaScore,
        #     "model_args": {
        #         "model_name": "cross-encoder/nli-roberta-base",
        #         "labels": ["contradiction", "entailment", "neutral"],
        #     },
        #     "premise_key": "split",
        # },
    }

    judge_llm_aggrefact(
        judge_model=judge_model,
        input_path="data/llm_aggrefact/preprocessed/cnn_human.json",
        output_path="data/llm_aggrefact/eval_cnn",
    )

    judge_llm_aggrefact(
        judge_model=judge_model,
        input_path="data/llm_aggrefact/preprocessed/Reveal.json",
        output_path="data/llm_aggrefact/eval_reveal",
    )

    judge_wiki_hallu(
        judge_model,
        input_path="data/wiki_bio_hallu/wiki_bio_simple.json",
        output_base_path="data/wiki_bio_hallu/eval_simple",
    )
    judge_wiki_hallu(
        judge_model,
        input_path="data/wiki_bio_hallu/wiki_bio_hard.json",
        output_base_path="data/wiki_bio_hallu/eval_hard",
    )
    judge_wiki_en(
        judge_model,
        input_path="data/wiki_en_sentence/preprocessed/test_data_pair_5000.json",
        output_base_path="data/wiki_en_sentence/eval_wiki_en",
    )
