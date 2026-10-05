"""
Fixtures partagées pour tous les tests CLIP LoRA Studio.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from modules.models import ModelRegistry
from modules.retrieval import ImageRetriever
from modules.analytics import Analytics
from utils.constants import CLASSES, EXAMPLES_DIR


# =============================================================================
# FIXTURES DE SESSION (chargées une seule fois)
# =============================================================================

@pytest.fixture(scope="session")
def registry():
    """ModelRegistry chargé une fois pour toute la session."""
    reg = ModelRegistry()
    reg.load_all()
    return reg


@pytest.fixture(scope="session")
def retriever():
    """ImageRetriever chargé une fois pour toute la session."""
    return ImageRetriever(
        embeddings_map={
            "Zero-shot": "assets/data/embeddings/clip_image_zs.npy",
        },
        labels_path="assets/data/embeddings/eval_labels.npy",
        images_dir="assets/eval_images",
    )


@pytest.fixture(scope="session")
def analytics():
    """Analytics chargé une fois pour toute la session."""
    return Analytics()


@pytest.fixture(scope="session")
def project_root():
    """Racine du projet."""
    return Path(__file__).parent.parent


# =============================================================================
# FIXTURES D'IMAGES
# =============================================================================

@pytest.fixture(scope="session")
def sample_image_paths():
    """Retourne 10 images d'exemple (1 par classe)."""
    paths = []
    for cls in CLASSES:
        # Essaie plusieurs formats de nom
        candidates = [
            EXAMPLES_DIR / f"{cls.lower().replace('/', '_')}.png",
            EXAMPLES_DIR / f"{cls}.png",
            EXAMPLES_DIR / f"{cls.lower()}.png",
        ]
        for c in candidates:
            if c.exists():
                paths.append((str(c), cls))
                break
    return paths


@pytest.fixture
def sample_image():
    """Retourne une image PIL de test (une sneaker si dispo)."""
    for name in ["sneaker.png", "Sneaker.png", "sneaker_0.png"]:
        p = EXAMPLES_DIR / name
        if p.exists():
            return Image.open(p).convert("RGB")
    # Fallback : génère une image grise
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_embedding():
    """Retourne un embedding 512D normalisé aléatoire."""
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


# =============================================================================
# SKIP CONDITIONS
# =============================================================================

def pytest_configure(config):
    """Marqueurs personnalisés."""
    config.addinivalue_line("markers", "slow: tests lents (> 5s)")
    config.addinivalue_line("markers", "gpu: tests nécessitant GPU")