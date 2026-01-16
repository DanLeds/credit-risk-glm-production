"""
Credit Scoring GLM Model Production
===================================
A production-ready framework for GLM model selection, training, and serving.
"""

from .glm_model import (
    GLMModelSelector,
    ModelConfig,
    ModelMetrics,
    ModelResult,
    ModelSelectionStrategy,
    ModelServing,
    DataValidator
)

__version__ = "1.0.0"
__all__ = [
    "GLMModelSelector",
    "ModelConfig",
    "ModelMetrics",
    "ModelResult",
    "ModelSelectionStrategy",
    "ModelServing",
    "DataValidator"
]
