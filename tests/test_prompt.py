from docqa.chunking import Chunk
from docqa.prompt import build_prompt


def test_build_prompt():
    question = "What is X?"
    chunk1 = Chunk(text="alpha text", source="doc.pdf", page=1)
    chunk2 = Chunk(text="beta text", source="doc.pdf", page=2)
    chunks = [chunk1, chunk2]
    prompt = build_prompt(question, chunks)
    expected = (
        "Answer the question using only the context below. "
        'If the context does not contain the answer, say "I don\'t know". '
        "Cite the sources you use in the form [source, p.N].\n"
        "\n"
        "Context:\n"
        "[doc.pdf, p.1]\n"
        "alpha text\n"
        "\n"
        "[doc.pdf, p.2]\n"
        "beta text\n"
        "\n"
        "Question: What is X?"
    )
    assert prompt == expected
