"""
Package modules — Modules métier (modèles, inférence, analytics).
"""
from .models import ModelRegistry
from .inference import classify, get_image_embedding, get_text_embeddings
from .analytics import Analytics

__all__ = ["ModelRegistry", "classify", "get_image_embedding",
           "get_text_embeddings", "Analytics"]