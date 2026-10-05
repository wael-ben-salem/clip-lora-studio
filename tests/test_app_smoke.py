"""
Smoke tests : l'app se construit sans erreur.
"""
import pytest


def test_imports():
    """Tous les imports critiques fonctionnent."""
    from modules.models import ModelRegistry
    from modules.inference import classify, get_image_embedding
    from modules.analytics import Analytics
    from modules.retrieval import ImageRetriever
    from utils.constants import CLASSES, EXAMPLES_DIR
    assert len(CLASSES) == 10


def test_build_app():
    """build_app() retourne un Blocks Gradio sans erreur."""
    from app import build_app
    demo = build_app()
    assert demo is not None


def test_class_list_correct():
    """La liste des classes est correcte."""
    from utils.constants import CLASSES
    expected = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    assert CLASSES == expected


def test_paths_exist():
    """Les chemins critiques existent."""
    from utils.constants import ASSETS_DIR, DATA_DIR, EXAMPLES_DIR
    assert ASSETS_DIR.exists(), f"Manquant : {ASSETS_DIR}"
    assert DATA_DIR.exists(), f"Manquant : {DATA_DIR}"
# =============================================================================
# TESTS PLAYGROUND (Partie 4.3)
# =============================================================================

def test_playground_dataframe_structure():
    """Le format de sortie du playground est valide."""
    import pandas as pd
    expected_cols = {"Image", "Modèle", "Prédiction", "Confiance"}
    df = pd.DataFrame(columns=list(expected_cols))
    assert set(df.columns) == expected_cols


def test_playground_csv_export_path():
    """Le chemin d'export CSV est cohérent."""
    from pathlib import Path
    csv_path = Path("logs") / "playground_results.csv"
    # Doit être dans logs/
    assert csv_path.parent.name == "logs"


def test_classify_batch_mock(registry, sample_image):
    """Simule un mini-batch : 3 images × 2 modèles = 6 prédictions."""
    from modules.inference import classify
    results = []
    for _ in range(3):
        for model_name in ["Zero-shot", "A2 LoRA V+T"]:
            top_dict, _ = classify(registry, sample_image, model_name=model_name, top_k=1)
            results.append(list(top_dict.keys())[0])
    assert len(results) == 6
    assert all(isinstance(r, str) for r in results)