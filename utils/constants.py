"""
Constantes partagées pour CLIP LoRA Studio.
"""

import torch


# =============================================================================
# CLASSES FASHION-MNIST
# =============================================================================
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

CLASS_PROMPTS = [f"a photo of a {c.lower()}" for c in CLASSES]


# =============================================================================
# PROMPTS HORS-DISTRIBUTION (pour oubli catastrophique)
# =============================================================================
OOD_PROMPTS = {
    "animaux":  ["a photo of a cat", "a photo of a dog", "a photo of a bird",
                 "a photo of a horse", "a photo of a fish"],
    "objets":   ["a photo of a car", "a photo of a phone", "a photo of a chair",
                 "a photo of a book", "a photo of a cup"],
    "formes":   ["a photo of a circle", "a photo of a square",
                 "a photo of a triangle", "a photo of a star"],
    "couleurs": ["a red object", "a blue object", "a green object",
                 "a yellow object", "a black object"],
}


# =============================================================================
# MODÈLE CLIP
# =============================================================================
CLIP_MODEL = "openai/clip-vit-base-patch32"


# =============================================================================
# DEVICE (auto-détection)
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CHEMINS
# =============================================================================
from pathlib import Path

ROOT_DIR         = Path(__file__).parent.parent
CHECKPOINTS_DIR  = ROOT_DIR / "checkpoints"
ASSETS_DIR       = ROOT_DIR / "assets"
FIGURES_DIR      = ASSETS_DIR / "figures"
DATA_DIR         = ASSETS_DIR / "data"
EXAMPLES_DIR     = ASSETS_DIR / "examples"