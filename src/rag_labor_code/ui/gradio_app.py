import logging

import gradio as gr

from rag_labor_code.pipeline.rag_pipeline import RAGPipeline
from rag_labor_code.ui.presenter import present_pipeline_result

LOGGER = logging.getLogger(__name__)

RAG_PIPELINE_TYPE = RAGPipeline

APP_TITLE = "Нейро-юрист по Трудовому кодексу РФ"

APP_DESCRIPTION = (
    "Задайте вопрос по трудовому праву Российской Федерации. "
    "Система найдёт релевантные положения Трудового кодекса, "
    "сформирует ответ и покажет использованные источники."
)

EMPTY_QUESTION_MESSAGE = "Введите вопрос по трудовому праву."
PROCESSING_ERROR_MESSAGE = "Во время обработки запроса произошла внутренняя ошибка."

ERROR_STATUS = "Ошибка"


def handle_question(
    question: str,
    pipeline: RAGPipeline,
) -> tuple[str, str, str]:
    if not isinstance(question, str):
        raise TypeError("question должен быть строкой!")

    if not isinstance(pipeline, RAG_PIPELINE_TYPE):
        raise TypeError("pipeline должен быть объектом RAGPipeline!")

    question = question.strip()

    if not question:
        return (
            EMPTY_QUESTION_MESSAGE,
            "",
            ERROR_STATUS,
        )

    try:
        result = pipeline.answer(question)

        return present_pipeline_result(result)

    except Exception:
        LOGGER.exception("Ошибка при обработке вопроса через RAG pipeline.")

        return (
            PROCESSING_ERROR_MESSAGE,
            "",
            ERROR_STATUS,
        )


def create_gradio_app(
    pipeline: RAGPipeline,
) -> gr.Blocks:
    if not isinstance(pipeline, RAG_PIPELINE_TYPE):
        raise TypeError("pipeline должен быть объектом RAGPipeline!")

    def submit_question(
        question: str,
    ) -> tuple[str, str, str]:
        return handle_question(
            question=question,
            pipeline=pipeline,
        )

    with gr.Blocks(
        title=APP_TITLE,
    ) as demo:
        gr.Markdown(f"# ⚖️ {APP_TITLE}")

        gr.Markdown(APP_DESCRIPTION)

        question_input = gr.Textbox(
            label="Ваш вопрос",
            placeholder=(
                "Например: " "какова максимальная продолжительность " "рабочей недели?"
            ),
            lines=3,
            max_lines=8,
            autofocus=True,
        )

        with gr.Row():
            submit_button = gr.Button(
                "Получить ответ",
                variant="primary",
            )

            clear_button = gr.Button(
                "Очистить",
            )

        status_output = gr.Textbox(
            label="Статус",
            interactive=False,
        )

        gr.Markdown("## Ответ")

        answer_output = gr.Markdown()

        gr.Markdown("## Источники")

        sources_output = gr.Markdown()

        outputs = [
            answer_output,
            sources_output,
            status_output,
        ]

        submit_button.click(
            fn=submit_question,
            inputs=question_input,
            outputs=outputs,
            api_name="ask",
            concurrency_limit=1,
            concurrency_id="rag_pipeline",
        )

        question_input.submit(
            fn=submit_question,
            inputs=question_input,
            outputs=outputs,
            api_name=False,
            concurrency_limit=1,
            concurrency_id="rag_pipeline",
        )

        clear_button.click(
            fn=lambda: (
                "",
                "",
                "",
                "",
            ),
            inputs=[],
            outputs=[
                question_input,
                answer_output,
                sources_output,
                status_output,
            ],
            queue=False,
        )

    demo.queue(
        max_size=8,
        default_concurrency_limit=1,
    )

    return demo


def launch_gradio_app(
    pipeline: RAGPipeline,
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    demo = create_gradio_app(
        pipeline=pipeline,
    )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
