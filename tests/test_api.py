import requests
from fastapi.testclient import TestClient

from docqa.api import create_app


def make_client(answer_fn=None):
    calls = []

    def default_answer(question):
        calls.append(question)
        return "the answer"

    app = create_app(answer_fn or default_answer)
    return TestClient(app), calls


def test_health():
    client, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_the_answer():
    client, calls = make_client()
    response = client.post("/ask", json={"question": "What is X?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "the answer"}
    assert calls == ["What is X?"]


def test_question_is_passed_unchanged():
    client, calls = make_client()
    client.post("/ask", json={"question": "  What?  "})
    assert calls == ["  What?  "]


def test_empty_and_blank_questions_are_rejected():
    client, calls = make_client()
    for bad in ("", "   ", "\n\t"):
        response = client.post("/ask", json={"question": bad})
        assert response.status_code == 422
    assert calls == []


def test_missing_or_invalid_field_is_rejected():
    client, calls = make_client()
    bodies = ({}, {"question": None}, {"question": ["a"]}, {"other": "x"})
    for body in bodies:
        assert client.post("/ask", json=body).status_code == 422
    assert calls == []


def test_question_length_limit():
    client, calls = make_client()
    ok = client.post("/ask", json={"question": "x" * 2000})
    too_long = client.post("/ask", json={"question": "x" * 2001})
    assert ok.status_code == 200
    assert too_long.status_code == 422
    assert calls == ["x" * 2000]


def test_llm_failure_returns_502():
    errors = (
        requests.exceptions.ConnectionError("down"),
        requests.exceptions.Timeout("slow"),
    )
    for error in errors:

        def broken(question, error=error):
            raise error

        client, _ = make_client(broken)
        response = client.post("/ask", json={"question": "What?"})
        assert response.status_code == 502
        assert response.json() == {
            "detail": "The language model is unavailable."
        }
