import argparse

from collections.abc import Sequence
from pathlib import Path

from rag_labor_code.bootstrap import build_rag_pipeline
from rag_labor_code.config import AppConfig
from rag_labor_code.ui.gradio_app import launch_gradio_app


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-labor-code",
        description=(
            "Локальная RAG-система для вопросов "
            "по Трудовому кодексу Российской Федерации."
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Путь к локальной Saiga-модели в формате GGUF.",
    )

    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=Path("data/raw/labor_code_rf.pdf"),
        help="Путь к PDF Трудового кодекса РФ.",
    )

    parser.add_argument(
        "--vector-index-dir",
        type=Path,
        default=Path("data/processed/vector_index"),
        help="Директория сохранённого векторного индекса.",
    )

    parser.add_argument(
        "--nemo-config-dir",
        type=Path,
        default=Path("configs/nemo"),
        help="Директория конфигурации NeMo Guardrails.",
    )

    parser.add_argument(
        "--embedding-device",
        type=str,
        default=None,
        help=("Устройство для embedding-модели, " "например cpu или cuda."),
    )

    parser.add_argument(
        "--reranker-device",
        type=str,
        default=None,
        help=("Устройство для CrossEncoder reranker, " "например cpu или cuda."),
    )

    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Размер контекстного окна Saiga.",
    )

    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help=(
            "Количество слоёв Saiga для GPU. "
            "-1 означает попытку выгрузить все слои на GPU."
        ),
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Количество CPU-потоков для llama.cpp.",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Принудительно пересоздать векторный индекс.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Адрес Gradio-сервера.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Порт Gradio-сервера.",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать публичную Gradio-ссылку.",
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = build_argument_parser()

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_arguments(argv)

    config = AppConfig(
        saiga_model_path=args.model_path,
        pdf_path=args.pdf_path,
        vector_index_dir=args.vector_index_dir,
        nemo_config_dir=args.nemo_config_dir,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        rebuild_index=args.rebuild_index,
    )

    pipeline = build_rag_pipeline(config)

    launch_gradio_app(
        pipeline=pipeline,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
