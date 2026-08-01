from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from sentence_transformers import CrossEncoder
from llama_cpp import Llama

from dataclasses import dataclass

from rag_labor_code.guardrails.nemo_guardrails import NemoGuardrailsAdapter
from rag_labor_code.generation.context_builder import ContextSource
from rag_labor_code.generation.context_builder import build_context
from rag_labor_code.guardrails.rules import check_query_guardrails
from rag_labor_code.guardrails.rules import check_answer_guardrails
from rag_labor_code.retrieval.hybrid_retriever import retrieve_hybrid_nodes
from rag_labor_code.reranking.cross_encoder import rerank_nodes
from rag_labor_code.generation.saiga_generator import generate_answer

NO_CONTEXT_ANSWER = (
    "Не удалось найти релевантные положения " "Трудового кодекса для ответа."
)

NEMO_GUARDRAILS_TYPE = NemoGuardrailsAdapter


@dataclass(frozen=True)
class RAGPipelineConfig:
    max_query_chars: int = 2_000
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
    reranker_batch_size: int = 8
    max_context_chars: int = 12_000
    max_sources: int = 5
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    require_sources: bool = True

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "max_query_chars": self.max_query_chars,
            "retrieval_top_k": self.retrieval_top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
            "reranker_batch_size": self.reranker_batch_size,
            "max_context_chars": self.max_context_chars,
            "max_sources": self.max_sources,
            "max_tokens": self.max_tokens,
        }

        for field_name, value in positive_integer_fields.items():
            if type(value) is not int:
                raise TypeError(f"{field_name} должен быть int!")

            if value <= 0:
                raise ValueError(f"{field_name} должен быть больше 0!")

        if type(self.temperature) is not float:
            raise TypeError("temperature должен быть float!")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature должна находиться от 0 до 2!")

        if type(self.top_p) is not float:
            raise TypeError("top_p должен быть float!")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p должен быть больше 0 и не больше 1!")

        if type(self.require_sources) is not bool:
            raise TypeError("require_sources должен быть bool!")


@dataclass(frozen=True)
class RAGPipelineResult:
    answer: str
    sources: tuple[ContextSource, ...]
    blocked: bool
    reason: str | None = None


class RAGPipeline:
    def __init__(
        self,
        index: VectorStoreIndex,
        bm25_retriever: BM25Retriever,
        reranker: CrossEncoder,
        llm: Llama,
        config: RAGPipelineConfig | None = None,
        nemo_guardrails: NemoGuardrailsAdapter | None = None,
    ) -> None:
        self._index = index
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._llm = llm

        if config is None:
            config = RAGPipelineConfig()
        else:
            if not isinstance(config, RAGPipelineConfig):
                raise TypeError("config должен быть объектом RAGPipelineConfig!")

        self._config = config

        if nemo_guardrails is not None and not isinstance(
            nemo_guardrails, NEMO_GUARDRAILS_TYPE
        ):
            raise TypeError(
                "nemo_guardrails должен быть объектом NemoGuardrailsAdapter!"
            )

        self._nemo_guardrails = nemo_guardrails

    def answer(
        self,
        question: str,
    ) -> RAGPipelineResult:

        query_decision = check_query_guardrails(
            query=question,
            max_chars=self._config.max_query_chars,
        )

        if not query_decision.allowed:
            return RAGPipelineResult(
                answer="",
                sources=(),
                blocked=True,
                reason=query_decision.reason,
            )

        effective_question = question

        if self._nemo_guardrails is not None:
            nemo_input_decision = self._nemo_guardrails.check_input(question)

            if not nemo_input_decision.allowed:
                return RAGPipelineResult(
                    answer="",
                    sources=(),
                    blocked=True,
                    reason=nemo_input_decision.reason,
                )

            effective_question = nemo_input_decision.content

        hybrid_candidates = retrieve_hybrid_nodes(
            index=self._index,
            bm25_retriever=self._bm25_retriever,
            query=effective_question,
            vector_top_k=self._config.retrieval_top_k,
            final_top_k=self._config.retrieval_top_k,
            rrf_k=self._config.rrf_k,
        )

        reranked_candidates = rerank_nodes(
            query=effective_question,
            candidates=hybrid_candidates,
            reranker=self._reranker,
            top_k=self._config.rerank_top_k,
            batch_size=self._config.reranker_batch_size,
        )

        context_result = build_context(
            candidates=reranked_candidates,
            max_sources=self._config.max_sources,
            max_chars=self._config.max_context_chars,
        )

        if not context_result.context.strip():
            return RAGPipelineResult(
                answer=NO_CONTEXT_ANSWER,
                sources=(),
                blocked=False,
                reason=None,
            )

        answer = generate_answer(
            question=effective_question,
            context=context_result.context,
            llm=self._llm,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )

        answer_decision = check_answer_guardrails(
            answer=answer,
            source_count=len(context_result.sources),
            require_sources=self._config.require_sources,
        )

        if not answer_decision.allowed:
            return RAGPipelineResult(
                answer="",
                sources=tuple(context_result.sources),
                blocked=True,
                reason=answer_decision.reason,
            )

        final_answer = answer

        if self._nemo_guardrails is not None:
            nemo_output_decision = self._nemo_guardrails.check_output(
                question=effective_question,
                answer=answer,
            )

            if not nemo_output_decision.allowed:
                return RAGPipelineResult(
                    answer="",
                    sources=tuple(context_result.sources),
                    blocked=True,
                    reason=nemo_output_decision.reason,
                )

            final_answer = nemo_output_decision.content

            if nemo_output_decision.modified:
                modified_answer_decision = check_answer_guardrails(
                    answer=final_answer,
                    source_count=len(context_result.sources),
                    require_sources=self._config.require_sources,
                )

                if not modified_answer_decision.allowed:
                    return RAGPipelineResult(
                        answer="",
                        sources=tuple(context_result.sources),
                        blocked=True,
                        reason=modified_answer_decision.reason,
                    )

        return RAGPipelineResult(
            answer=final_answer,
            sources=tuple(context_result.sources),
            blocked=False,
            reason=None,
        )
