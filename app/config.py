from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    lmstudio_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "qwen3.5-0.8b")
    # lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "qwen3.5-9b")

    
    lmstudio_api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    memory_path: str = os.getenv("MEMORY_PATH", "data/memory.json")

settings = Settings()