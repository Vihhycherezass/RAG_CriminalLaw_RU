from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType, RailsResult

from rag_labor_code.guardrails.llama_cpp_nemo_model import (
    LlamaCppNeMoModel,
    DEFAULT_NEMO_MODEL_NAME,
)

NEMO_INPUT_BLOCK_REASON = "NeMo Guardrails заблокировал входной запрос."

NEMO_OUTPUT_BLOCK_REASON = "NeMo Guardrails заблокировал ответ модели."

LLM_RAILS_TYPE = LLMRails


@dataclass(frozen=True)
class NemoGuardrailDecision:
    allowed: bool
    content: str
    modified: bool
    reason: str | None = None
    rail: str | None = None


class NemoGuardrailsAdapter:
    def __init__(
        self,
        rails: LLMRails,
    ) -> None:

        if not isinstance(rails, LLM_RAILS_TYPE):
            raise TypeError("rails должен быть объектом LLMRails!")

        self._rails = rails

    def _build_decision(
        self,
        result: RailsResult,
        original_content: str,
        blocked_reason: str,
    ) -> NemoGuardrailDecision:

        status = result.status

        if status == RailStatus.BLOCKED:
            return NemoGuardrailDecision(
                allowed=False,
                content="",
                modified=False,
                reason=blocked_reason,
                rail=result.rail,
            )

        elif status == RailStatus.PASSED:
            return NemoGuardrailDecision(
                allowed=True,
                content=original_content,
                modified=False,
            )

        elif status == RailStatus.MODIFIED:
            if not isinstance(result.content, str) or not result.content.strip():
                raise ValueError(
                    "Изменённый NeMo Guardrails контент не должен быть пустым!"
                )

            content = result.content.strip()

            return NemoGuardrailDecision(
                allowed=True,
                content=content,
                modified=True,
            )

        else:
            raise ValueError("NeMo Guardrails вернул неизвестный статус!")

    def check_input(
        self,
        query: str,
    ) -> NemoGuardrailDecision:

        if not isinstance(query, str):
            raise TypeError("query должен быть строкой!")

        query = query.strip()

        if not query:
            raise ValueError("query не должен быть пустым!")

        result = self._rails.check(
            messages=[
                {
                    "role": "user",
                    "content": query,
                },
            ],
            rail_types=[RailType.INPUT],
        )

        decision = self._build_decision(
            result=result,
            original_content=query,
            blocked_reason=NEMO_INPUT_BLOCK_REASON,
        )

        return decision

    def check_output(
        self,
        question: str,
        answer: str,
    ) -> NemoGuardrailDecision:

        if not isinstance(question, str):
            raise TypeError("question должен быть строкой!")

        question = question.strip()

        if not question:
            raise ValueError("question не должен быть пустым!")

        if not isinstance(answer, str):
            raise TypeError("answer должен быть строкой!")

        answer = answer.strip()

        if not answer:
            raise ValueError("answer не должен быть пустым!")

        result = self._rails.check(
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ],
            rail_types=[RailType.OUTPUT],
        )

        decision = self._build_decision(
            result=result,
            original_content=answer,
            blocked_reason=NEMO_OUTPUT_BLOCK_REASON,
        )

        return decision


def create_nemo_guardrails_adapter(
    config_path: str | Path,
    llm: Llama,
    model_name: str = DEFAULT_NEMO_MODEL_NAME,
) -> NemoGuardrailsAdapter:

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError("Директория конфигурации NeMo не найдена!")

    if not config_path.is_dir():
        raise ValueError("config_path должен указывать на директорию!")

    config = RailsConfig.from_path(str(config_path))

    nemo_model = LlamaCppNeMoModel(llm=llm, model_name=model_name)

    rails = LLMRails(
        config=config,
        llm=nemo_model,
    )

    return NemoGuardrailsAdapter(rails)
