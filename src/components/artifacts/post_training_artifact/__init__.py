"""Post-training-artifact stage facade."""

from .core import materialize_post_trained_artifact
from .materializer import materialize_post_training_artifact

__all__ = [
    "materialize_post_trained_artifact",
    "materialize_post_training_artifact",
]
