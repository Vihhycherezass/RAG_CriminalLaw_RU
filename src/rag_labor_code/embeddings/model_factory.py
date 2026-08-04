from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DEFAULT_E5_MODEL_NAME = "intfloat/multilingual-e5-large"


def create_e5_embed_model(
    model_name: str = DEFAULT_E5_MODEL_NAME,
    device: str | None = None,
    max_length: int = 512,
    embed_batch_size: int = 16,
) -> BaseEmbedding:
    """Создает локальную embedding-модель multilingual E5."""

    if not model_name or not model_name.strip():
        raise ValueError("Название embedding-модели не должно быть пустым!")

    if max_length <= 0:
        raise ValueError("max_length должен быть больше 0!")

    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size должен быть больше 0!")

    if device is not None and not device.strip():
        raise ValueError("Название устройства не должно быть пустым!")

    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=max_length,
        normalize=True,
        embed_batch_size=embed_batch_size,
        device=device,
        show_progress_bar=False,
    )

    return embed_model
