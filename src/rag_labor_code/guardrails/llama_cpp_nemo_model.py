from collections.abc import AsyncIterator
from typing import Any
import asyncio

from llama_cpp import Llama
from nemoguardrails.types import ChatMessage, LLMResponse, LLMResponseChunk

LLAMA_TYPE = Llama

DEFAULT_NEMO_MODEL_NAME = "saiga-mistral-7b-gguf"


class LlamaCppNeMoModel:
    def __init__(
        self,
        llm: Llama,
        model_name: str = DEFAULT_NEMO_MODEL_NAME,
    ) -> None:
        if not isinstance(llm, LLAMA_TYPE):
            raise TypeError("llm должен быть объектом Llama!")

        if not isinstance(model_name, str):
            raise TypeError("model_name должен быть str!")

        model_name = model_name.strip()

        if not model_name:
            raise ValueError("model_name не должен быть пустым!")

        self._llm = llm
        self._model_name = model_name

    def _prepare_message(self, prompt: str | list[ChatMessage]) -> list[dict[str, str]]:

        if isinstance(prompt, str):
            prompt = prompt.strip()

            if not prompt:
                raise ValueError("prompt не должен быть пустым!")

            return [{"role": "user", "content": prompt}]

        if not isinstance(prompt, list):
            raise TypeError("prompt должен быть строкой или списком ChatMessage!")

        if not prompt:
            raise ValueError("Список сообщений не должен быть пустым!")

        messages: list[dict[str, str]] = []

        for message in prompt:
            if not isinstance(message, ChatMessage):
                raise TypeError("Каждый элемент prompt должен быть ChatMessage!")

            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError("Содержимое ChatMessage должно быть непустой строкой!")

            content = message.content.strip()

            role = message.role.value

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return messages

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "llama_cpp"

    @property
    def provider_url(self) -> None:
        return None

    async def generate_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:

        messages = self._prepare_message(prompt)

        generation_kwargs = dict(kwargs)
        generation_kwargs["stream"] = False

        if stop is not None:
            generation_kwargs["stop"] = stop

        response = await asyncio.to_thread(
            self._llm.create_chat_completion,
            messages=messages,
            **generation_kwargs,
        )

        if not isinstance(response, dict):
            raise ValueError("Некорректный ответ llama.cpp!")

        choices = response.get("choices")

        if not isinstance(choices, list) or not choices:
            raise ValueError("Некорректный ответ llama.cpp!")

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise ValueError("Некорректный ответ llama.cpp!")

        message = first_choice.get("message")

        if not isinstance(message, dict):
            raise ValueError("Некорректный ответ llama.cpp!")

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Некорректный ответ llama.cpp!")

        finish_reason = first_choice.get("finish_reason")

        if finish_reason is None:
            finish_reason = "stop"

        allowed_finish_reasons = {
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "error",
            "other",
        }

        if finish_reason not in allowed_finish_reasons:
            finish_reason = "other"

        return LLMResponse(
            content=content.strip(),
            model=self._model_name,
            finish_reason=finish_reason,
        )

    async def stream_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:

        response = await self.generate_async(
            prompt,
            stop=stop,
            **kwargs,
        )

        yield LLMResponseChunk(
            delta_content=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
        )
