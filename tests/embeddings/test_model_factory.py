from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_labor_code.embeddings.model_factory import (
    DEFAULT_E5_MODEL_NAME,
    create_e5_embed_model,
)


@patch("rag_labor_code.embeddings.model_factory.HuggingFaceEmbedding")
def test_create_e5_embed_model_creates_huggingface_model(
    mock_huggingface_embedding: MagicMock,
) -> None:
    expected_model = MagicMock(spec=BaseEmbedding)
    mock_huggingface_embedding.return_value = expected_model

    model = create_e5_embed_model()

    mock_huggingface_embedding.assert_called_once_with(
        model_name=DEFAULT_E5_MODEL_NAME,
        max_length=512,
        normalize=True,
        embed_batch_size=16,
        device=None,
        show_progress_bar=False,
    )

    assert model is expected_model


@patch("rag_labor_code.embeddings.model_factory.HuggingFaceEmbedding")
def test_create_e5_embed_model_passes_custom_parameters(
    mock_huggingface_embedding: MagicMock,
) -> None:
    expected_model = MagicMock(spec=BaseEmbedding)
    mock_huggingface_embedding.return_value = expected_model

    model = create_e5_embed_model(
        model_name="custom/e5-model",
        device="cuda",
        max_length=256,
        embed_batch_size=32,
    )

    mock_huggingface_embedding.assert_called_once_with(
        model_name="custom/e5-model",
        max_length=256,
        normalize=True,
        embed_batch_size=32,
        device="cuda",
        show_progress_bar=False,
    )

    assert model is expected_model


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_create_e5_embed_model_rejects_empty_model_name(
    model_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Название embedding-модели не должно быть пустым!",
    ):
        create_e5_embed_model(model_name=model_name)


@pytest.mark.parametrize(
    "max_length",
    [
        0,
        -1,
    ],
)
def test_create_e5_embed_model_rejects_invalid_max_length(
    max_length: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_length должен быть больше 0!",
    ):
        create_e5_embed_model(max_length=max_length)


@pytest.mark.parametrize(
    "embed_batch_size",
    [
        0,
        -1,
    ],
)
def test_create_e5_embed_model_rejects_invalid_batch_size(
    embed_batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="embed_batch_size должен быть больше 0!",
    ):
        create_e5_embed_model(
            embed_batch_size=embed_batch_size,
        )


def test_create_e5_embed_model_rejects_empty_device() -> None:
    with pytest.raises(
        ValueError,
        match="Название устройства не должно быть пустым!",
    ):
        create_e5_embed_model(device="   ")
