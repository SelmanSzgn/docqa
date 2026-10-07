import pytest

from docqa.config import get_required_env


def test_get_required_env_returns_value(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "abc123")
    assert get_required_env("LLM_API_KEY") == "abc123"


def test_get_required_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing"):
        get_required_env("LLM_API_KEY")


def test_get_required_env_raises_when_blank(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", " ")
    with pytest.raises(RuntimeError, match="blank"):
        get_required_env("LLM_API_KEY")


def test_get_required_env_raises_when_empty(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="empty"):
        get_required_env("LLM_API_KEY")
