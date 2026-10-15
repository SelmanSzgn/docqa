import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(max_length=2000)


def create_app(answer_fn, sources_fn=None) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AnswerRequest):
        question = req.question
        if not question.strip():
            raise HTTPException(
                status_code=422, detail="question must not be blank"
            )
        try:
            answer = answer_fn(question)
        except requests.exceptions.RequestException:
            raise HTTPException(
                status_code=502, detail="The language model is unavailable."
            ) from None
        return {"answer": answer}

    if sources_fn is not None:

        @app.post("/ask/sources")
        def sources(req: AnswerRequest):
            question = req.question
            if not question.strip():
                raise HTTPException(
                    status_code=422, detail="question must not be blank"
                )
            try:
                out = sources_fn(question)
            except requests.exceptions.RequestException:
                raise HTTPException(
                    status_code=502,
                    detail="The language model is unavailable.",
                ) from None
            return out

    return app
