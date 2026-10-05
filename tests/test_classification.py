"""
Tests pour modules/inference.py — Classification.
"""
import pytest
import pandas as pd
from PIL import Image

from modules.inference import classify, classify_multi, get_image_embedding
from utils.constants import CLASSES


def test_classify_returns_dict_and_df(registry, sample_image):
    """classify() retourne un dict + DataFrame."""
    top_dict, df = classify(registry, sample_image, model_name="Zero-shot", top_k=5)
    assert isinstance(top_dict, dict)
    assert isinstance(df, pd.DataFrame)


def test_classify_top_k_size(registry, sample_image):
    """Le dict top contient k entrées."""
    top_dict, df = classify(registry, sample_image, model_name="Zero-shot", top_k=5)
    assert len(top_dict) == 5


def test_classify_df_has_all_classes(registry, sample_image):
    """Le DataFrame contient les 10 classes."""
    top_dict, df = classify(registry, sample_image, model_name="Zero-shot", top_k=5)
    assert len(df) == 10, f"Attendu 10 lignes, obtenu {len(df)}"


def test_classify_probabilities_sum_to_1(registry, sample_image):
    """La somme des probabilités ≈ 1.0."""
    top_dict, df = classify(registry, sample_image, model_name="Zero-shot", top_k=10)
    total = df["Probability"].sum()
    assert abs(total - 1.0) < 0.01, f"Somme = {total}, attendu ≈ 1.0"


def test_classify_all_models(registry, sample_image):
    """Les 3 modèles donnent des résultats."""
    for model_name in registry.names:
        top_dict, df = classify(registry, sample_image, model_name=model_name, top_k=3)
        assert len(top_dict) == 3
        assert all(cls in CLASSES for cls in top_dict.keys())


def test_classify_none_raises(registry):
    """classify(None) lève une erreur propre."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        classify(registry, None, model_name="Zero-shot", top_k=5)


def test_get_image_embedding_shape(registry, sample_image):
    """L'embedding a la bonne shape."""
    emb = get_image_embedding(registry, "Zero-shot", sample_image)
    # emb peut être torch tensor ou numpy
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    emb = emb.flatten()
    assert emb.shape[-1] == 512, f"Embedding shape : {emb.shape}"


def test_get_image_embedding_normalized(registry, sample_image):
    """L'embedding image est normalisé."""
    import numpy as np
    emb = get_image_embedding(registry, "Zero-shot", sample_image)
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    emb = emb.flatten()
    norm = np.linalg.norm(emb)
    # Tolérance large : CLIP normalise généralement
    assert 0.9 <= norm <= 1.1, f"Norme = {norm}"
    
# =============================================================================
# TESTS COMPARATEUR (Partie 4.1)
# =============================================================================

def test_classify_multi_returns_all_models(registry, sample_image):
    """classify_multi() retourne une ligne par modèle."""
    df = classify_multi(registry, sample_image, top_k=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(registry.names), (
        f"Attendu {len(registry.names)} modèles, obtenu {len(df)}"
    )
    assert "Model" in df.columns
    assert "Prediction" in df.columns
    assert "Confidence" in df.columns


def test_classify_multi_consensus(registry, sample_image):
    """Sur une image simple, les modèles devraient être d'accord."""
    df = classify_multi(registry, sample_image, top_k=1)
    predictions = df["Prediction"].tolist()
    # Au moins 2 modèles sur 3 doivent être d'accord (tolérance)
    from collections import Counter
    most_common_count = Counter(predictions).most_common(1)[0][1]
    assert most_common_count >= 2, (
        f"Aucun consensus : {predictions}"
    )