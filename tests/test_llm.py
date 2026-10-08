from docqa import llm


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "Bonjour!"}


def test_generate_returns_response_text(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.generate("Say hello in French.") == "Bonjour!"


def test_generate_sends_model_and_prompt(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["json"] = json
        return FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    llm.generate("my prompt")
    assert sent["url"] == llm.URL
    assert sent["json"] == {
        "model": llm.MODEL,
        "prompt": "my prompt",
        "stream": False,
    }
