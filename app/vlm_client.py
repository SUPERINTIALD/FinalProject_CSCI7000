from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib import response

import requests

from app.config import settings


class VLMClient:
    """
    Separate VLM client.

    This is intentionally separate from LMStudioClient so the planner can stay
    on Qwen 0.8B while perception uses a different vision-language model.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (base_url or settings.vlm_base_url).rstrip("/")
        self.model = model or settings.vlm_model
        self.api_key = api_key or settings.vlm_api_key
        self.timeout = timeout

    @staticmethod
    def _image_to_data_url(image_path: str | Path) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "image/png"

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def generate_from_image(
        self,
        *,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        image_url = self._image_to_data_url(image_path)

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        # response.raise_for_status()
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            print("\n[VLM HTTP ERROR]")
            print("Status:", response.status_code)
            print("Body:", response.text)
            raise exc

        data = response.json()
        return data["choices"][0]["message"]["content"]