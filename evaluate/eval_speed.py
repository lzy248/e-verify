import os, sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.append(os.path.split(sys.path[0])[0])
from model.Decomposer import *
from util import *
import time
from model.Embedding import *
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error
from eval import eval
import numpy as np


def decompose(path, decomposer: Decomposer):
    data = read_json(path)

    texts = [d["bio"] for d in data]
    start = time.time()
    decomposed = decomposer.decompose(texts)
    decom_time = time.time() - start

    for i, d in enumerate(data):
        d["decomposed"] = decomposed[i]

    return data, {"decom_time": decom_time}


def fact_score_X(
    premise: List[str],
    hypothesis: List[str],
    model: PairScore,
) -> Tuple[List[float], float]:

    all_pairs = [(p, h) for h in hypothesis for p in premise]

    start = time.time()
    all_scores = model.get_score(all_pairs)
    score_time = time.time() - start

    max_scores = []
    index = 0
    for h in hypothesis:
        num_premises = len(premise)
        hypothesis_scores = all_scores[index : index + num_premises]
        max_scores.append(max(hypothesis_scores))
        index += num_premises

    assert len(max_scores) == len(hypothesis)
    return max_scores, score_time


def embedding_text(
    texts: List[str], embedder: DeEmbedder
) -> Tuple[torch.Tensor, float]:
    start = time.time()
    embed = embedder.get_embedding(texts, verbose=True)
    embed_time = time.time() - start
    return embed.to("cpu"), embed_time


def fact_score_E(
    premise_embed: torch.Tensor,  # [batch_size, (...,) dim]
    hypothesis_embed: torch.Tensor,  # [batch_size, (...,) dim]
    sim_calculator: DeSim,
) -> Tuple[List[float], float]:
    assert (
        premise_embed.shape[1:] == hypothesis_embed.shape[1:]
    ), "Inner dimensions of premise and hypothesis must match."

    batch_premise = premise_embed.unsqueeze(0)
    batch_hypothesis = hypothesis_embed.unsqueeze(1)

    premise_repeat_dims = [1] * batch_premise.dim()
    hypothesis_repeat_dims = [1] * batch_hypothesis.dim()

    premise_repeat_dims[0] = len(hypothesis_embed)
    hypothesis_repeat_dims[1] = len(premise_embed)

    batch_premise = batch_premise.repeat(*premise_repeat_dims)
    batch_hypothesis = batch_hypothesis.repeat(*hypothesis_repeat_dims)

    flat_premise = batch_premise.view(-1, *premise_embed.shape[1:])
    flat_hypothesis = batch_hypothesis.view(-1, *premise_embed.shape[1:])

    start_sim = time.time()
    sim_scores = sim_calculator(flat_premise, flat_hypothesis)
    sim_time = time.time() - start_sim

    sim_scores = sim_scores.view(len(hypothesis_embed), len(premise_embed))
    max_similarities = sim_scores.max(dim=1).values.tolist()

    return max_similarities, sim_time


def evaluate_decompose(path, out_path, decomposer: Decomposer):
    # path = "data/wiki_bio_hallu/model_gen"
    # out_path = "data/wiki_bio_hallu/model_gen/decomposed_qwen"
    os.makedirs(out_path, exist_ok=True)
    log_dict = {}
    for data_file in os.listdir(path):
        name = data_file.split(".")[0]
        if not data_file.endswith(".json"):
            continue
        logging.info(f"Decomposing {data_file}")
        data, time_log = decompose(f"{path}/{data_file}", decomposer)
        log_dict[data_file] = time_log
        write_json(data, f"{out_path}/{name}.json")
    log_dict["Total_decompose_time"] = sum([v["decom_time"] for v in log_dict.values()])
    write_json(log_dict, f"{out_path}/time.log")
    logging.info(
        f"Total decompose time: {log_dict['Total_decompose_time']:.2f} seconds"
    )


def embed_decompose(path, out_path, embedder):
    # path = "data/wiki_bio_hallu/model_gen/decomposed_qwen"
    # out_path = "data/wiki_bio_hallu/model_gen/embed"
    os.makedirs(out_path, exist_ok=True)
    log_dict = {}

    for data_file in os.listdir(path):
        name = data_file.split(".")[0]
        if not data_file.endswith(".json"):
            continue
        logging.info(f"Embedding {data_file}")
        data = read_json(f"{path}/{data_file}")
        texts = [d["decomposed"] for d in data]
        lens = [len(t) for t in texts]
        flat_texts = [t for text in texts for t in text]
        embed, embed_time = embedding_text(flat_texts, embedder)

        out_embed = []
        index = 0
        for l in lens:
            out_embed.append(embed[index : index + l])
            index += l

        # pickle.dump(out_embed, open(f"{out_path}/{name}.pkl", "wb"))
        torch.save(out_embed, f"{out_path}/{name}.pt")
        log_dict[name] = {"embed_time": embed_time}

    write_json(log_dict, f"{out_path}/time.log")


def evaluate_X(path, out_file, model: PairScore, premise_key="bio_split"):

    threshold = 0.5
    log_dict = {}

    wiki_data = read_json("data/wiki_bio_hallu/model_gen/data/wiki.json")
    for data_file in os.listdir(path):
        name = data_file.split(".")[0]
        if not data_file.endswith(".json") or data_file == "wiki.json":
            continue
        data = read_json(f"{path}/{data_file}")
        total_time = 0
        scores = []
        for pre, hyp in tqdm(zip(wiki_data, data), postfix=name, total=len(data)):
            premise = pre[premise_key]
            hypothesis = hyp["decomposed"]
            if isinstance(hypothesis, str):
                hypothesis = [hypothesis]
            if isinstance(premise, str):
                premise = [premise]
            if len(hypothesis) == 0:
                scores.append({"score_list": [], "fact_score": 0})
                continue
            s, score_time = fact_score_X(premise, hypothesis, model)
            total_time += score_time

            scores.append(
                {
                    "score_list": s,
                    "fact_score": float(sum([1 for i in s if i > threshold]) / len(s)),
                }
            )
        log_dict[name] = {
            "total_time": total_time,
            "scores": scores,
            "avg_score": float(sum([s["fact_score"] for s in scores]) / len(scores)),
        }

    write_json(log_dict, out_file)


def evaluate_E(path, out_file, sim_calculator: DeSim):

    threshold = 0.5
    log_dict = {}

    wiki_embed = torch.load(f"{path}/wiki.pt")
    for data_file in os.listdir(path):
        name = data_file.split(".")[0]
        if not data_file.endswith(".pt") or data_file == "wiki.pt":
            continue
        data = torch.load(f"{path}/{data_file}")
        total_time = 0
        scores = []
        for pre, hyp in tqdm(zip(wiki_embed, data), postfix=name, total=len(data)):
            if len(hyp) == 0:
                scores.append({"score_list": [], "fact_score": 0})
                continue
            s, score_time = fact_score_E(pre.to(device), hyp.to(device), sim_calculator)
            total_time += score_time
            scores.append(
                {
                    "score_list": s,
                    "fact_score": float(sum([1 for i in s if i > threshold]) / len(s)),
                }
            )
        log_dict[name] = {
            "total_time": total_time,
            "scores": scores,
            "avg_score": float(sum([s["fact_score"] for s in scores]) / len(scores)),
        }

    write_json(log_dict, out_file)


def corr_fact_score(path, anchor_file):
    anchor_data = read_json(anchor_file)
    keylist = list(anchor_data.keys())
    anchor_scores = []
    for key in keylist:
        for item in anchor_data[key]["scores"]:
            anchor_scores.append(item["fact_score"])

    result = {}
    for data_file in os.listdir(path):
        if not data_file.endswith(".json"):
            continue
        name = data_file.split(".")[0]
        data = read_json(f"{path}/{data_file}")
        data_scores = []
        for key in keylist:
            for item in data[key]["scores"]:
                data_scores.append(item["fact_score"])

        pearson_corr, _ = pearsonr(anchor_scores, data_scores)
        mae = mean_absolute_error(anchor_scores, data_scores)

        mean_score = np.mean(data_scores)
        std_dev = np.std(data_scores)
        cv = std_dev / mean_score if mean_score != 0 else 0

        result[name] = {
            "pearson": pearson_corr,
            "mae": mae,
            "cv": cv,
        }

    sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[0])}

    write_json(sorted_result, f"{path}/corr_scores.log")


def qwen0_5_de():
    evaluate_decompose(
        path="data/wiki_bio_hallu/model_gen",
        out_path="data/wiki_bio_hallu/model_gen/decomposed_qwen0_5b",
        decomposer=QwenSingleDecomposer(
            model_name="experiment/ckpt/decomposer-qwen2.5-0-5b",
            batch_size=32,
            vllm_boost=True,
        ),
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_qwen0_5b",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_qwen0_5b_qwen2.5_7b.json",
        model=LLMScore(model_name="Qwen/Qwen2.5-7B-Instruct", vllm_boost=True),
        premise_key="bio",
    )
    embed_decompose(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_qwen0_5b",
        out_path="data/wiki_bio_hallu/model_gen/embed/embed_qwen0_5_d_p_lora",
        embedder=DeEmbedder.load_from_ckpt("experiment/ckpt/d_p_lora").to(device),
    )
    evaluate_E(
        path="data/wiki_bio_hallu/model_gen/embed/embed_qwen0_5_d_p_lora",
        out_file="data/wiki_bio_hallu/model_gen/scores/E_score_qwen0_5_d_p_lora.json",
        sim_calculator=DeSim.load_from_ckpt("experiment/ckpt/d_p_lora").to(device),
    )


def gpt4o_de():
    evaluate_decompose(
        path="data/wiki_bio_hallu/model_gen",
        out_path="data/wiki_bio_hallu/model_gen/decomposed_gpt4o",
        decomposer=APIDecomposer(api_name="gpt-4o"),
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/decomposed_gpt4o",
        out_path="data/wiki_bio_hallu/model_gen/X_score_gpt4o_deberta",
        model=DeBERTaScore(
            model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            labels=["entailment", "neutral", "contradiction"],
        ),
    )

    embed_decompose(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_gpt4o",
        out_path="data/wiki_bio_hallu/model_gen/embed/embed_gpt4o_d_p_lora",
        embedder=DeEmbedder.load_from_ckpt("experiment/ckpt/d_p_lora", n_latest=3).to(
            device
        ),
    )
    evaluate_E(
        path="data/wiki_bio_hallu/model_gen/embed/embed_gpt4o_d_p_lora",
        out_file="data/wiki_bio_hallu/model_gen/scores/E_score_gpt4o_d_p_lora.json",
        sim_calculator=DeSim.load_from_ckpt("experiment/ckpt/d_p_lora", n_latest=3).to(
            device
        ),
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_gpt4o",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_gpt4o_qwen2.5_7b.json",
        model=LLMScore(model_name="Qwen/Qwen2.5-7B-Instruct", vllm_boost=True),
        premise_key="bio",
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/decomposed_gpt4o",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_gpt4o_gpt4o.json",
        model=APIScore(api_name="gpt-4o"),
        premise_key="bio",
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_gpt4o",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_gpt4o_minicheck_2.json",
        model=DeBERTaScore(
            model_name="lytang/MiniCheck-DeBERTa-v3-Large",
            labels=["contradiction", "entailment"],
        ),
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_gpt4o",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_gpt4o_minicheck_flan-t5_2",
        model=FlanT5Score(model_name="lytang/MiniCheck-Flan-T5-Large"),
    )

    evaluate_X(
        path="data/wiki_bio_hallu/model_gen/deco/decomposed_gpt4o",
        out_file="data/wiki_bio_hallu/model_gen/scores/X_score_gpt4o_nli-roberta_2.json",
        model=DeBERTaScore(
            model_name="cross-encoder/nli-roberta-base",
            labels=["contradiction", "entailment", "neutral"],
        ),
    )


if __name__ == "__main__":
    # qwen0_5_de()
    # gpt4o_de()
    corr_fact_score(
        path="data/wiki_bio_hallu/model_gen/scores",
        anchor_file="data/wiki_bio_hallu/model_gen/gen_label/human_score.json",
    )
