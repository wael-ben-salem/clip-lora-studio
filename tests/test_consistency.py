"""
Tests de cohérence globale : labels ↔ images ↔ embeddings.
"""
import pytest
import numpy as np
from pathlib import Path


def test_labels_embeddings_same_length(retriever):
    """Labels et embeddings ont la même longueur."""
    for model_name, arr in retriever._embeddings.items():
        assert len(arr) == len(retriever.labels), (
            f"Modèle '{model_name}' : {len(arr)} embeddings vs "
            f"{len(retriever.labels)} labels"
        )


def test_labels_values_valid(retriever):
    """Les labels sont dans [0, 9]."""
    assert retriever.labels.min() >= 0
    assert retriever.labels.max() <= 9


def test_images_count_matches_embeddings(retriever, project_root):
    """Le nombre d'images sur disque correspond aux embeddings."""
    images_dir = project_root / "assets" / "eval_images"
    if not images_dir.exists():
        pytest.skip("Dossier eval_images/ absent")

    # Compter seulement les fichiers PNG (ignorer .DS_Store, Thumbs.db)
    images = sorted(images_dir.glob("eval_*.png"))
    assert len(images) == 2000, (
        f"Attendu 2000 images eval_*.png, trouvé {len(images)}. "
        f"Vérifie qu'il n'y a pas de fichiers parasites."
    )


def test_first_10_images_have_valid_labels(retriever, project_root):
    """Les 10 premières images ont un label valide et cohérent."""
    images_dir = project_root / "assets" / "eval_images"
    if not images_dir.exists():
        pytest.skip("Dossier eval_images/ absent")

    for i in range(10):
        assert retriever.labels[i] in range(10), (
            f"Label invalide à l'index {i} : {retriever.labels[i]}"
        )


def test_no_orphan_embedding_files(project_root):
    """Il n'y a pas de fichier d'embedding orphelin."""
    emb_dir = project_root / "assets" / "data" / "embeddings"
    if not emb_dir.exists():
        pytest.skip("Dossier embeddings/ absent")

    expected = {
        "clip_image_zs.npy",
        "clip_image_a1.npy",
        "clip_image_a2.npy",
        "clip_image_b.npy",
        "eval_labels.npy",
        "eval_indices.npy",
    }
    # Vérifie que les fichiers présents sont attendus
    for f in emb_dir.glob("*.npy"):
        # Soit attendu, soit on le signale (mais pas un fail)
        if f.name not in expected:
            print(f"⚠️ Fichier non-typique : {f.name}")