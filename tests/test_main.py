from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_labor_code import main as main_module
from rag_labor_code.main import (
    build_argument_parser,
    main,
    parse_arguments,
)


def test_build_argument_parser_returns_parser() -> None:
    parser = build_argument_parser()

    assert parser is not None


def test_parse_arguments_uses_defaults() -> None:
    args = parse_arguments(
        [
            "--model-path",
            "models/saiga.gguf",
        ]
    )

    assert args.model_path == Path("models/saiga.gguf")

    assert args.pdf_path == Path("data/raw/labor_code_rf.pdf")

    assert args.vector_index_dir == Path("data/processed/vector_index")

    assert args.nemo_config_dir == Path("configs/nemo")

    assert args.embedding_device is None
    assert args.reranker_device is None

    assert args.n_ctx == 8192
    assert args.n_gpu_layers == -1
    assert args.n_threads is None

    assert args.rebuild_index is False

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.share is False


def test_parse_arguments_accepts_custom_values() -> None:
    args = parse_arguments(
        [
            "--model-path",
            "models/custom.gguf",
            "--pdf-path",
            "data/custom.pdf",
            "--vector-index-dir",
            "indexes/custom",
            "--nemo-config-dir",
            "configs/custom_nemo",
            "--embedding-device",
            "cpu",
            "--reranker-device",
            "cpu",
            "--n-ctx",
            "4096",
            "--n-gpu-layers",
            "0",
            "--n-threads",
            "4",
            "--rebuild-index",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--share",
        ]
    )

    assert args.model_path == Path("models/custom.gguf")

    assert args.pdf_path == Path("data/custom.pdf")

    assert args.vector_index_dir == Path("indexes/custom")

    assert args.nemo_config_dir == Path("configs/custom_nemo")

    assert args.embedding_device == "cpu"
    assert args.reranker_device == "cpu"

    assert args.n_ctx == 4096
    assert args.n_gpu_layers == 0
    assert args.n_threads == 4

    assert args.rebuild_index is True

    assert args.host == "0.0.0.0"
    assert args.port == 8080
    assert args.share is True


def test_parse_arguments_requires_model_path() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])


def test_main_builds_pipeline_and_launches_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object()

    build_pipeline_mock = MagicMock(return_value=pipeline)

    launch_mock = MagicMock()

    monkeypatch.setattr(
        main_module,
        "build_rag_pipeline",
        build_pipeline_mock,
    )

    monkeypatch.setattr(
        main_module,
        "launch_gradio_app",
        launch_mock,
    )

    main(
        [
            "--model-path",
            "models/saiga.gguf",
            "--embedding-device",
            "cpu",
            "--reranker-device",
            "cpu",
            "--n-ctx",
            "4096",
            "--n-gpu-layers",
            "0",
            "--n-threads",
            "4",
            "--rebuild-index",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )

    build_pipeline_mock.assert_called_once()

    config = build_pipeline_mock.call_args.args[0]

    assert config.saiga_model_path == Path("models/saiga.gguf")

    assert config.embedding_device == "cpu"
    assert config.reranker_device == "cpu"

    assert config.n_ctx == 4096
    assert config.n_gpu_layers == 0
    assert config.n_threads == 4

    assert config.rebuild_index is True

    launch_mock.assert_called_once_with(
        pipeline=pipeline,
        server_name="127.0.0.1",
        server_port=9000,
        share=False,
    )


def test_main_passes_share_to_gradio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object()

    monkeypatch.setattr(
        main_module,
        "build_rag_pipeline",
        MagicMock(return_value=pipeline),
    )

    launch_mock = MagicMock()

    monkeypatch.setattr(
        main_module,
        "launch_gradio_app",
        launch_mock,
    )

    main(
        [
            "--model-path",
            "models/saiga.gguf",
            "--share",
        ]
    )

    assert launch_mock.call_args.kwargs["share"] is True
