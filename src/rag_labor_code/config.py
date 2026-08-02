from dataclasses import dataclass, field
from pathlib import Path

from rag_labor_code.pipeline.rag_pipeline import RAGPipelineConfig


@dataclass(frozen=True)
class AppConfig:
    saiga_model_path: Path

    pdf_path: Path = Path("data/raw/labor_code_rf.pdf")

    vector_index_dir: Path = Path("data/processed/vector_index")

    nemo_config_dir: Path = Path("configs/nemo")

    chunk_size: int = 512
    chunk_overlap: int = 100

    embedding_device: str | None = None
    reranker_device: str | None = None

    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_threads: int | None = None

    rebuild_index: bool = False

    pipeline_config: RAGPipelineConfig = field(default_factory=RAGPipelineConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.saiga_model_path, Path):
            raise TypeError("saiga_model_path должен быть Path!")

        if not isinstance(self.pdf_path, Path):
            raise TypeError("pdf_path должен быть Path!")

        if not isinstance(self.vector_index_dir, Path):
            raise TypeError("vector_index_dir должен быть Path!")

        if not isinstance(self.nemo_config_dir, Path):
            raise TypeError("nemo_config_dir должен быть Path!")

        if type(self.chunk_size) is not int:
            raise TypeError("chunk_size должен быть int!")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size должен быть больше 0!")

        if type(self.chunk_overlap) is not int:
            raise TypeError("chunk_overlap должен быть int!")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap не должен быть меньше 0!")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size!")

        if self.embedding_device is not None:
            if not isinstance(self.embedding_device, str):
                raise TypeError("embedding_device должен быть str или None!")

            if not self.embedding_device.strip():
                raise ValueError("embedding_device не должен быть пустым!")

        if self.reranker_device is not None:
            if not isinstance(self.reranker_device, str):
                raise TypeError("reranker_device должен быть str или None!")

            if not self.reranker_device.strip():
                raise ValueError("reranker_device не должен быть пустым!")

        if type(self.n_ctx) is not int:
            raise TypeError("n_ctx должен быть int!")

        if self.n_ctx <= 0:
            raise ValueError("n_ctx должен быть больше 0!")

        if type(self.n_gpu_layers) is not int:
            raise TypeError("n_gpu_layers должен быть int!")

        if self.n_gpu_layers < -1:
            raise ValueError("n_gpu_layers должен быть не меньше -1!")

        if self.n_threads is not None:
            if type(self.n_threads) is not int:
                raise TypeError("n_threads должен быть int или None!")

            if self.n_threads <= 0:
                raise ValueError("n_threads должен быть больше 0!")

        if type(self.rebuild_index) is not bool:
            raise TypeError("rebuild_index должен быть bool!")

        if not isinstance(
            self.pipeline_config,
            RAGPipelineConfig,
        ):
            raise TypeError(
                "pipeline_config должен быть объектом " "RAGPipelineConfig!"
            )
