from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

from src.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Groq-backed LLM wrapper with deterministic defaults and structured output."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.groq_model
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @staticmethod
    def _base_url() -> str:
        return "https://api.groq.com/openai/v1"

    @property
    def client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError(
                "Groq API key is not set. Add GROQ_API_KEY to .env."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self._base_url())
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        if not self.api_key:
            raise RuntimeError(
                "Groq API key is not set. Add GROQ_API_KEY to .env."
            )
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self._base_url())
        return self._async_client

    def _completion_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": settings.temperature,
            "seed": settings.groq_seed,
        }

    @staticmethod
    def _quota_error_message() -> str:
        return (
            "Groq is rate-limited or out of quota. Check your API key, billing, "
            "or use an offline command such as eval --offline."
        )

    def complete_json(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                **self._completion_kwargs(),
            )
        except RateLimitError as exc:
            raise RuntimeError(self._quota_error_message()) from exc
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return schema.model_validate(data)

    def stream_text(self, system: str, user: str) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                **self._completion_kwargs(),
            )
        except RateLimitError as exc:
            raise RuntimeError(self._quota_error_message()) from exc
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def astream_text(self, system: str, user: str) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                **self._completion_kwargs(),
            )
        except RateLimitError as exc:
            raise RuntimeError(self._quota_error_message()) from exc
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
