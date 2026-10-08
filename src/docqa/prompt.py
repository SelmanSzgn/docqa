from docqa.chunking import Chunk


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    blocks = [f"[{c.source}, p.{c.page}]\n{c.text}" for c in chunks]
    context = "\n\n".join(blocks)
    return (
        "Answer the question using only the context below. "
        'If the context does not contain the answer, say "I don\'t know".'
        "Cite the sources you use in the form [source, p.N].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
