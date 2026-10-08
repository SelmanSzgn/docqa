import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True)
