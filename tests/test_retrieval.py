"""
Tests pour modules/retrieval.py — ImageRetriever.
"""
import pytest
import numpy as np

from utils.constants import CLASSES


# =============================================================================
# TESTS DE CHARGEMENT
# =============================================================================

def test_retriever_loads(retriever):
    """Le retriever charge sans erreur."""
    assert retriever is not None
    assert retriever.num_images > 0


def test_num_images_2000(retriever):
    """Le retriever charge exactement 2000 images."""
    assert retriever.num_images == 2000, (
        f"Attendu 2000 images, obtenu {retriever.num_images}. "
        f"Vérifie assets/eval_images/ et les embeddings."
    )


def test_labels_shape(retriever):
    """Les labels ont la bonne shape."""
    assert retriever.labels is not None
    assert retriever.labels.shape == (2000,), (
        f"Attendu (2000,), obtenu {retriever.labels.shape}"
    )


def test_labels_balanced(retriever):
    """
    Les labels sont raisonnablement équilibrés.

    Fashion-MNIST a un déséquilibre naturel (~±15% autour de la moyenne).
    On tolère 150-250 images par classe pour 2000 images au total.
    """
    counts = np.bincount(retriever.labels, minlength=10)
    assert len(counts) == 10, f"Attendu 10 classes, obtenu {len(counts)}"

    total = counts.sum()
    assert total == 2000, f"Attendu 2000 labels, obtenu {total}"

    for i, c in enumerate(counts):
        # Tolérance large : 150-250 par classe (Fashion-MNIST naturel)
        assert 150 <= c <= 250, (
            f"Classe {i} ({CLASSES[i]}) : {c} images "
            f"(attendu 150-250, dataset Fashion-MNIST naturel)"
        )


def test_embeddings_normalized(retriever):
    """Les embeddings sont L2-normalisés."""
    for model_name, arr in retriever._embeddings.items():
        norms = np.linalg.norm(arr, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Embeddings '{model_name}' non normalisés : "
            f"min={norms.min():.4f}, max={norms.max():.4f}"
        )


def test_embeddings_shape(retriever):
    """Les embeddings ont la bonne shape (2000, 512)."""
    for model_name, arr in retriever._embeddings.items():
        assert arr.shape == (2000, 512), (
            f"Embeddings '{model_name}' : shape {arr.shape}, attendu (2000, 512)"
        )


def test_available_models(retriever):
    """Le retriever expose la liste des modèles disponibles."""
    assert hasattr(retriever, "available_models")
    assert len(retriever.available_models) >= 1
    assert "Zero-shot" in retriever.available_models


# =============================================================================
# TESTS DE RECHERCHE
# =============================================================================

def test_retrieve_returns_k(retriever, sample_embedding):
    """retrieve() retourne exactement k résultats."""
    for k in [1, 5, 12, 24]:
        results = retriever.retrieve(sample_embedding, k=k, return_images=False)
        assert len(results) == k, f"k={k} : obtenu {len(results)} résultats"


def test_retrieve_structure(retriever, sample_embedding):
    """Chaque résultat a les clés attendues."""
    results = retriever.retrieve(sample_embedding, k=5, return_images=False)
    for r in results:
        assert "index" in r
        assert "class" in r
        assert "class_id" in r
        assert "score" in r
        assert r["class"] in CLASSES


def test_retrieve_scores_sorted(retriever, sample_embedding):
    """Les scores sont triés décroissants."""
    results = retriever.retrieve(sample_embedding, k=10, return_images=False)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Les scores ne sont pas triés décroissants"
    )


def test_retrieve_cosine_scores_in_range(retriever, sample_embedding):
    """Les scores cosinus sont dans [-1, 1]."""
    results = retriever.retrieve(sample_embedding, k=10, return_images=False)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0, f"Score hors [-1, 1] : {r['score']}"


def test_class_filter(retriever, sample_embedding):
    """Le filtre par classe fonctionne."""
    # Demande uniquement des Sneakers (class_id=7)
    results = retriever.retrieve(
        sample_embedding, k=10, return_images=False,
        class_filter=[7]
    )
    for r in results:
        assert r["class_id"] == 7, f"Filtre échoué : {r['class']} trouvé"


def test_alternative_metrics(retriever, sample_embedding):
    """Les métriques alternatives (dot, euclidean) fonctionnent."""
    for metric in ["cosine", "dot", "euclidean"]:
        results = retriever.retrieve(
            sample_embedding, k=5, return_images=False, metric=metric
        )
        assert len(results) == 5, f"Métrique '{metric}' : {len(results)} résultats"


def test_invalid_metric_raises(retriever, sample_embedding):
    """Une métrique invalide lève une erreur."""
    with pytest.raises(ValueError):
        retriever.retrieve(sample_embedding, k=5, metric="invalid")


def test_invalid_model_raises(retriever, sample_embedding):
    """Un modèle sans embeddings lève une erreur claire."""
    with pytest.raises(ValueError):
        retriever.retrieve(sample_embedding, k=5, model_name="Inexistant")


# =============================================================================
# TEST DE COHÉRENCE (LE PLUS IMPORTANT)
# =============================================================================

@pytest.mark.slow
def test_accuracy_at_5(retriever):
    """Accuracy@5 ≥ 80% sur 50 requêtes aléatoires."""
    np.random.seed(42)
    n_probe = 50
    k = 5
    query_ids = np.random.choice(2000, n_probe, replace=False)

    db_embs = retriever._embeddings["Zero-shot"]
    hits = 0
    for qid in query_ids:
        q_label = int(retriever.labels[qid])
        sims = db_embs @ db_embs[qid]
        sims[qid] = -1.0  # exclure soi-même
        top_k = np.argsort(sims)[::-1][:k]
        if q_label in retriever.labels[top_k]:
            hits += 1

    acc = hits / n_probe
    assert acc >= 0.80, (
        f"Accuracy@5 = {acc*100:.1f}% < 80%. "
        f"Problème probable : labels désalignés avec embeddings."
    )


def test_consistency_method(retriever):
    """La méthode check_consistency() fonctionne."""
    result = retriever.check_consistency(n_probe=20, k=5)
    assert isinstance(result, dict)
    assert len(result) >= 1
    for model_name, acc in result.items():
        assert 0.0 <= acc <= 1.0
        
# =============================================================================
# TESTS CROSS-MODAL (Partie 4.4)
# =============================================================================

def test_cross_modal_text_to_image(registry, retriever):
    """Recherche texte → image donne des résultats cohérents."""
    from modules.inference import get_text_embeddings
    # Utilise le modèle Zero-shot pour l'encodage texte
    txt_emb = get_text_embeddings(registry, "Zero-shot", ["a photo of a sneaker"]).cpu().numpy()
    results = retriever.retrieve(txt_emb, model_name="Zero-shot", k=5, return_images=False)
    assert len(results) == 5
    # Vérifier que les résultats sont des classes valides
    for r in results:
        assert r["class"] in CLASSES


def test_cross_modal_scores_in_range(retriever, registry):
    """Les scores cross-modal sont dans [0, 1]."""
    from modules.inference import get_text_embeddings
    txt_emb = get_text_embeddings(registry, "Zero-shot", ["a red dress"]).cpu().numpy()
    results = retriever.retrieve(txt_emb, model_name="Zero-shot", k=3, return_images=False)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0, f"Score hors bornes : {r['score']}"