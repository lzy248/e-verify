import os, sys

sys.path.append(os.path.split(sys.path[0])[0])

from util import *


def eval_wiki_en(
    eval_file: str = None,
    eval_result_path="data/wiki_en_sentence/eval_5000/",
    freeze_threshold=0.5,
):
    print(eval_result_path)
    for file in os.listdir(eval_result_path):
        if eval_file and eval_file != file:
            continue
        if file.endswith(".json"):
            res = read_json(os.path.join(eval_result_path, file))
            y_true = [str.lower(d["label"]) == "entailment" for d in res]
            y_pred = [
                not any(score < freeze_threshold for score in d["score"]) for d in res
            ]
            eval_res = eval(file, y_pred, y_true, print_result=False)
            print(f"model: {file:<40}", end=" ")
            print(f"acc: {eval_res['acc']:.4f}", end=" ")
            print(f"f1: {eval_res['f1']:.4f}", end=" ")
            print(f"recall: {eval_res['recall']:.4f}", end=" ")
            print(f"precision: {eval_res['precision']:.4f}")
    eval("All true", [True] * len(y_true), y_true)
    eval("All false", [False] * (len(y_true)), y_true)
    eval("Random", [random.choice([True, False]) for _ in range(len(y_true))], y_true)


def eval_wiki_hallu(
    eval_file: str = None,
    freeze_threshold=0.5,
    label_file: str = "data/wiki_bio_gpt3_hallucination/wiki_bio_simple.json",
    eval_result_path: str = "data/wiki_bio_gpt3_hallucination/eval_simple/",
):
    print(f"path: {eval_result_path}")
    print("========wiki_bio_hallucination========")
    label_data = read_json(label_file)
    labels = [label == "yes" for d in label_data for label in d["labels"]]
    for file in sorted(os.listdir(eval_result_path)):
        if eval_file and eval_file != file:
            continue
        if file.endswith(".json"):
            file_path = os.path.join(eval_result_path, file)
            res = read_json(file_path)
            if "score" in res[0].keys():
                scores = []
                for d in res:
                    scores.extend(d["score"])

                y_pred = [score >= freeze_threshold for score in scores]
                eval_res = eval(file, y_pred, labels, print_result=False)
                print(f"model: {file:<40}", end=" ")
                print(f"acc: {eval_res['acc']:.4f}", end=" ")
                print(f"f1: {eval_res['f1']:.4f}", end=" ")
                print(f"recall: {eval_res['recall']:.4f}", end=" ")
                print(f"precision: {eval_res['precision']:.4f}")
            else:
                pred = [p == "yes" for d in res for p in d["pred"]]  # llm checker
                eval(file, pred, labels)

    eval("All true", [True] * len(labels), labels)
    eval("All false", [False] * (len(labels)), labels)
    eval("Random", [random.choice([True, False]) for _ in range(len(labels))], labels)


def eval_llm_aggrefact(
    eval_file: str = None,
    freeze_threshold=0.5,
    label_file: str = "data/llm_aggrefact/preprocessed/aggrefact_cnn.json",
    eval_result_path: str = "data/llm_aggrefact/eval_cnn/",
):
    print(f"path: {eval_result_path}")
    print("========llm_aggrefact========")
    label_data = read_json(label_file)
    labels = [label == 1 for d in label_data for label in d["score"]]
    for file in sorted(os.listdir(eval_result_path)):
        if eval_file and eval_file != file:
            continue
        if file.endswith(".json"):
            file_path = os.path.join(eval_result_path, file)
            res = read_json(file_path)
            if "score" in res[0].keys():
                scores = []
                for d in res:
                    scores.extend(d["score"])
                y_pred = [score >= freeze_threshold for score in scores]
                eval_res = eval(file, y_pred, labels, print_result=False)
                print(f"model: {file:<40}", end=" ")
                print(f"acc: {eval_res['acc']:.4f}", end=" ")
                print(f"f1: {eval_res['f1']:.4f}", end=" ")
                print(f"recall: {eval_res['recall']:.4f}", end=" ")
                print(f"precision: {eval_res['precision']:.4f}")
            else:
                pred = [p == "yes" for d in res for p in d["pred"]]  # llm checker
                eval(file, pred, labels)
    eval("All true", [True] * len(labels), labels)
    eval("All false", [False] * (len(labels)), labels)
    eval("Random", [random.choice([True, False]) for _ in range(len(labels))], labels)


if __name__ == "__main__":

    eval_llm_aggrefact(
        eval_result_path="data/llm_aggrefact/eval_cnn/",
        label_file="data/llm_aggrefact/preprocessed/cnn_human.json",
    )
    eval_llm_aggrefact(
        eval_result_path="data/llm_aggrefact/eval_reveal",
        label_file="data/llm_aggrefact/preprocessed/reveal_human.json",
    )
    eval_wiki_en(eval_result_path="data/wiki_en_sentence/eval_5000")

    eval_wiki_hallu(
        eval_result_path="data/wiki_bio_gpt3_hallucination/eval_simple",
        label_file="data/wiki_bio_gpt3_hallucination/wiki_bio_simple.json",
    )
    eval_wiki_hallu(
        eval_result_path="data/wiki_bio_gpt3_hallucination/eval_hard",
        label_file="data/wiki_bio_gpt3_hallucination/wiki_bio_hard.json",
    )
