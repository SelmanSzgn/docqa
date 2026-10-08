import requests

MODEL = "qwen2.5:3b"
URL = "http://localhost:11434/api/generate"
TIMEOUT = 120


def generate(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    response = requests.post(URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data["response"]
