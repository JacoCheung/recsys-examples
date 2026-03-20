# Auto-register training pipelines when pipeline module is imported
from . import sw_train_pipeline  # noqa: F401
from . import train_pipeline  # noqa: F401
from .train_pipeline_factory import TrainPipelineFactory

__all__ = ["TrainPipelineFactory"]
