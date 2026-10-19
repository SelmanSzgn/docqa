import importlib

from docqa import llm

DEFAULT = "http://localhost:11434/api/generate"
DOCKER = "http://host.docker.internal:11434/api/generate"


def reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OLLAMA_URL", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_URL", value)
    importlib.reload(llm)
    return llm.URL


def check(monkeypatch, value, expected):
    try:
        assert reload_with(monkeypatch, value) == expected
    finally:
        # Remet l'environnement et le module dans leur état d'origine.
        monkeypatch.undo()
        importlib.reload(llm)


def test_default_url_when_variable_is_absent(monkeypatch):
    check(monkeypatch, None, DEFAULT)


def test_url_comes_from_the_variable(monkeypatch):
    check(monkeypatch, DOCKER, DOCKER)


def test_empty_variable_falls_back_to_default(monkeypatch):
    check(monkeypatch, "", DEFAULT)
