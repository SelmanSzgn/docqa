import os

from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    key = os.environ.get(name)
    if key is None:
        raise RuntimeError(
            f"{name} is missing: copy .env.example to .env and fill it in."
        )
    elif key == "":
        raise RuntimeError(
            f"{name} is empty: copy .env.example to .env and fill it in."
        )
    elif not key.strip():
        raise RuntimeError(
            f"{name} is blank: copy .env.example to .env and fill it in."
        )
    return key
