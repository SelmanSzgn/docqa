import pytest
import requests
from fastapi.testclient import TestClient

from docqa.api import create_app

RESULT = {
    "answer": "the answer",
    "sources": [{"source": "doc.pdf", "page": 4}],
}


def make_client(sources_fn=None):
    calls = {"answer": [], "sources": []}

    def answer_fn(question):
        calls["answer"].append(question)
        return "plain answer"

    def default_sources(question):
        calls["sources"].append(question)
        return RESULT

    app = create_app(answer_fn, sources_fn=sources_fn or default_sources)
    return TestClient(app), calls


def test_route_does_not_exist_without_sources_fn():
    client = TestClient(create_app(lambda q: "x"))
    response = client.post("/ask/sources", json={"question": "What?"})
    assert response.status_code == 404


def test_returns_the_dict_unchanged():
    client, _ = make_client()
    response = client.post("/ask/sources", json={"question": "What?"})
    assert response.status_code == 200
    assert response.json() == RESULT


def test_question_is_passed_unchanged_and_called_once():
    client, calls = make_client()
    client.post("/ask/sources", json={"question": "  What?  "})
    assert calls["sources"] == ["  What?  "]
    assert calls["answer"] == []


def test_invalid_questions_are_rejected():
    client, calls = make_client()
    bodies = (
        {"question": ""},
        {"question": "   "},
        {"question": "\n\t"},
        {},
        {"question": None},
        {"question": ["a"]},
        {"other": "x"},
    )
    for body in bodies:
        response = client.post("/ask/sources", json=body)
        assert response.status_code == 422
    assert calls["sources"] == []


def test_question_length_limit():
    client, calls = make_client()
    ok = client.post("/ask/sources", json={"question": "x" * 2000})
    too_long = client.post("/ask/sources", json={"question": "x" * 2001})
    assert ok.status_code == 200
    assert too_long.status_code == 422
    assert calls["sources"] == ["x" * 2000]


def test_llm_failure_returns_502():
    errors = (
        requests.exceptions.ConnectionError("down"),
        requests.exceptions.Timeout("slow"),
    )
    for error in errors:

        def broken(question, error=error):
            raise error

        client, _ = make_client(broken)
        response = client.post("/ask/sources", json={"question": "What?"})
        assert response.status_code == 502
        assert response.json() == {
            "detail": "The language model is unavailable."
        }


def test_other_exceptions_are_not_caught():
    def broken(question):
        raise RuntimeError("bug")

    client, _ = make_client(broken)
    with pytest.raises(RuntimeError, match="bug"):
        client.post("/ask/sources", json={"question": "What?"})


def test_ask_route_still_uses_answer_fn_only():
    client, calls = make_client()
    response = client.post("/ask", json={"question": "What?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "plain answer"}
    assert calls["answer"] == ["What?"]
    assert calls["sources"] == []
