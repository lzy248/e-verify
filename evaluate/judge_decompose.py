import os, sys

sys.path.append(os.path.split(sys.path[0])[0])

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from util import *
from model.Embedding import (
    compare_two_texts,
    PairScore,
    Embedding,
    DeBERTaScore,
    APIScore,
)
from tqdm import tqdm


def judge_decompose_precise(
    decomposed: List[str],
    anchor: List[str],
    judge_model: PairScore | Embedding,
    threshold: float = 0.5,
) -> float:
    """judge the decomposed result by comparing it with the reference texts

    Args:
        decomposed (List[str]): the decomposed texts to be judged
        anchor (List[str]): the reference texts
        judge_model (PairScore | Embedding): the judge model to compare the texts
        threshold (float, optional): the threshold to determine if the texts are similar.

    Returns:
        float: the precision score of the decomposed texts
    """
    anchor = [" ".join(anchor)]

    p_score = compare_two_texts(
        premise_decomposed=anchor,
        hypothesis_decomposed=decomposed,
        compare_fn=judge_model,
    )
    labels = [1 if s > threshold else 0 for s in p_score]
    correct = len([i for i in labels if i == 1])
    total = len(labels)
    return correct / total, correct, total


def combine_texts(texts: List[str], max_len: int = 256) -> List[str]:
    """combine the texts into texts with max length for efficient comparison

    Args:
        texts (List[str]): the texts to be combined
        max_len (int, optional): the max length of the combined text. Defaults to 256.

    Returns:
        List[str]: the combined text
    """
    combined = []
    combined_text = ""
    for text in texts:
        if len(combined_text) + len(text) < max_len:
            combined_text += " " + text
        else:
            combined.append(combined_text)
            combined_text = text
    return combined


def judge_decompose_recall(
    decomposed: List[str],
    oracle_decomposed: List[str],
    judge_model: PairScore | Embedding,
    threshold: float = 0.5,
) -> float:
    """judge the decomposed result by comparing it with the oracle decomposed texts to calculate the recall score

    Args:
        decomposed (List[str]): the decomposed texts to be judged
        oracle_decomposed (List[str]): the oracle decomposed texts to be compared with
        judge_model (PairScore | Embedding): the judge model to compare the texts
        threshold (float, optional): the threshold to determine if the texts are similar.

    Returns:
        float: the recall score of the decomposed texts
    """
    # combine the decomposed texts with max length for efficient comparison
    decomposed = [" ".join(decomposed)]

    p_score = compare_two_texts(
        premise_decomposed=decomposed,
        hypothesis_decomposed=oracle_decomposed,
        compare_fn=judge_model,
    )

    labels = [1 if s > threshold else 0 for s in p_score]
    correct = len([i for i in labels if i == 1])
    total = len(labels)
    return correct / total, correct, total


def judge_decompose(path):
    judge = APIScore()
    decompose = read_json(path)
    oracle_decompose = read_json(
        "data/wiki_bio_gpt3_hallucination/wiki_bio_simple.json"
    )
    recalls = []
    precisions = []
    recall_correct = 0
    recall_total = 0
    precise_correct = 0
    precise_total = 0
    for dec, orac in tqdm(zip(decompose, oracle_decompose), total=len(decompose)):
        recall, re_cor, re_tal = judge_decompose_recall(
            decomposed=dec["wiki_decomposed"],
            oracle_decomposed=orac["wiki_decomposed"],
            judge_model=judge,
        )
        recalls.append(recall)
        recall_correct += re_cor
        recall_total += re_tal
        precision, pre_cor, pre_total = judge_decompose_precise(
            decomposed=dec["wiki_decomposed"],
            anchor=orac["splitted_wiki"],
            judge_model=judge,
        )
        precisions.append(precision)
        precise_correct += pre_cor
        precise_total += pre_total

    # recall and precise
    print(f"Mean Recall: {sum(recalls) / len(recalls)}")
    print(f"Mean Precision: {sum(precisions) / len(precisions)}"),
    print(f"Recall Acc: {recall_correct/ recall_total}")
    print(f"Precision Acc: {precise_correct/ precise_total}")
    return recalls, precisions


if __name__ == "__main__":
    # judge_decompose("results/gpt4o_decompose.json") # Mean Recall: 0.9991379310344828 Mean Precision: 0.9830495030342901
    # judge_decompose(
    #     "results/t5_decompose.json"
    # )  # Mean Recall: 0.945979314988608 Mean Precision: 0.9512266246041301

    # judge_decompose(
    #     "experiment/results/qwen_full_decompose_ln2e5_bs_16.json"
    # )
    # ln2e5 Mean Recall: 0.972504654717859 Mean Precision: 0.9627780950022994 Recall Acc: 0.9692008429242989 Precision Acc: 0.9431336161187699

    judge_decompose("experiment/results/qwen_kd.json")
    # 7b Mean Recall: 0.9653713037744428 Mean Precision: 0.9777337860541209 Recall Acc: 0.9654725239098719 Precision Acc: 0.9736872609250029
    # 7b sft Mean Recall: 0.9794613794658299 Mean Precision: 0.9798588317922547 Recall Acc: 0.9786026908737234 Precision Acc: 0.9783436146475916

    pass
