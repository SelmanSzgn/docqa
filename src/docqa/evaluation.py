import json
from pathlib import Path


def hit_at_k(retrieved_pages: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be positive, k={k} was given.")
    if len(relevant) == 0:
        raise ValueError("relevant set must not be empty.")
    return float(any([p in retrieved_pages[:k] for p in relevant]))


def reciprocal_rank(retrieved_pages: list[int], relevant: set[int]) -> float:
    if len(relevant) == 0:
        raise ValueError("relevant set must not be empty")
    found = False
    for _rank, p in enumerate(retrieved_pages, start=1):
        if p in relevant:
            found = True
            break
    if found:
        return 1 / _rank
    else:
        return 0.0


def evaluate(retrieve, dataset: list[dict], k: int) -> dict[str, float]:
    if k <= 0:
        raise ValueError(f"k must be positive, k={k} was given.")
    if len(dataset) == 0:
        raise ValueError("dataset is empty.")
    n_questions = len(dataset)
    metrics = {"hit_rate": 0.0, "mrr": 0.0}
    for i in range(n_questions):
        question = dataset[i]["question"]
        retrieved_pages = retrieve(question)
        relevant = dataset[i]["relevant_pages"]

        metrics["hit_rate"] += hit_at_k(retrieved_pages, relevant, k)
        metrics["mrr"] += reciprocal_rank(retrieved_pages, relevant)
    metrics["hit_rate"] /= n_questions
    metrics["mrr"] /= n_questions
    return metrics


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("dataset is not a list.")
    if len(dataset) == 0:
        raise ValueError("dataset is an empty list.")
    for i in range(len(dataset)):
        if "question" not in dataset[i].keys():
            raise ValueError("An entry of the dataset has no question.")
        q = dataset[i]["question"]
        if not isinstance(q, str):
            raise ValueError("A question in the dataset is not a string.")
        if not q.strip():
            raise ValueError("Empty or blank question found in the dataset.")
        if "relevant_pages" not in dataset[i].keys():
            raise ValueError("An entry of the dataset has no relevant_pages.")
        relevant = dataset[i]["relevant_pages"]
        if not isinstance(relevant, list):
            raise ValueError("A relevant_pages of the dataset is not a list.")
        if len(relevant) == 0:
            raise ValueError("Empty relevant_pages found.")
        for j in relevant:
            if not isinstance(j, int):
                raise ValueError(
                    "All relevant_pages must contain integers only."
                )
            if j < 1:
                raise ValueError("Integers in relevant_pages must be >= 1.")
    return dataset
