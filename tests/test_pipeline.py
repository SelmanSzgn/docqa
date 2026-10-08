import numpy as np

from docqa import pipeline
from docqa.chunking import Chunk


def test_answer_uses_best_chunks(monkeypatch):
    chunks = [
        Chunk("alpha", "doc.pdf", 1),
        Chunk("beta", "doc.pdf", 2),
        Chunk("gamma", "doc.pdf", 3),
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    captured = {}

    def fake_embed(model, texts):
        return np.array([[1.0, 0.0]])

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "final answer"

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(pipeline, "generate", fake_generate)

    result = pipeline.answer("What?", chunks, vecs, model=None, k=2)

    assert result == "final answer"
    assert "alpha" in captured["prompt"]
    assert "gamma" in captured["prompt"]
    assert "beta" not in captured["prompt"]
    assert "Question: What?" in captured["prompt"]
