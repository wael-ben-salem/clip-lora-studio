"""
Tests pour modules/analytics.py — Analytics.
"""
import pytest
import pandas as pd
from pathlib import Path
from modules.analytics import Analytics
from modules.inference import classify


def test_team_table_not_empty(analytics):
    """team_table n'est pas vide."""
    assert not analytics.team_table.empty


def test_team_table_columns(analytics):
    """team_table contient les colonnes attendues."""
    cols = [c.lower() for c in analytics.team_table.columns]
    # Au moins une colonne de chaque type doit exister
    has_model = any("model" in c for c in cols)
    has_acc = any("acc" in c for c in cols)
    has_p1 = any("p1" in c or "p@1" in c for c in cols)
    assert has_model, f"Pas de colonne 'model' dans {list(analytics.team_table.columns)}"
    assert has_acc, f"Pas de colonne 'acc' dans {list(analytics.team_table.columns)}"
    assert has_p1, f"Pas de colonne 'p1' dans {list(analytics.team_table.columns)}"


def test_confusion_delta_not_empty(analytics):
    """confusion_delta n'est pas vide."""
    assert not analytics.confusion_delta.empty


def test_data_files_exist(analytics, project_root):
    """Les fichiers CSV existent dans assets/data/."""
    data_dir = project_root / "assets" / "data"
    assert data_dir.exists(), f"Dossier manquant : {data_dir}"
    csvs = list(data_dir.glob("*.csv"))
    assert len(csvs) >= 3, f"Attendu au moins 3 CSV, trouvé {len(csvs)}"


def test_get_figure_returns_path_or_none(analytics):
    """get_figure() retourne un chemin valide ou None."""
    result = analytics.get_figure("TEAM_FINAL_TABLE_heatmap.png", "team")
    # Soit un Path/str, soit None
    assert result is None or isinstance(result, (str, Path))


def test_list_available_figures(analytics):
    """list_available_figures() retourne une liste."""
    figures = analytics.list_available_figures("team")
    assert isinstance(figures, list)


def test_list_available_data(analytics):
    """list_available_data() retourne une liste."""
    data = analytics.list_available_data()
    assert isinstance(data, list)
    assert len(data) >= 3, f"Attendu au moins 3 fichiers, trouvé {len(data)}"


def test_robustness_attribute(analytics):
    """L'attribut robustness existe et est un DataFrame."""
    if hasattr(analytics, "robustness"):
        assert isinstance(analytics.robustness, pd.DataFrame)
# =============================================================================
# TESTS CALIBRATION (Partie 4.2)
# =============================================================================

def test_calibration_figures_exist(analytics):
    """Les figures de calibration doivent exister (ou être None proprement)."""
    fig_cal = analytics.get_figure("team_calibration_reliability.png", "team")
    fig_conf = analytics.get_figure("team_confidence_distribution.png", "team")
    # Au moins une des deux doit exister (le projet en génère au moins une)
    assert fig_cal is not None or fig_conf is not None, (
        "Aucune figure de calibration trouvée. "
        "Vérifie assets/figures/team/"
    )


def test_classify_returns_probability_confidence(registry, sample_image):
    """classify() retourne des probabilités interprétables."""
    top_dict, df = classify(registry, sample_image, model_name="Zero-shot", top_k=3)
    for cls, prob in top_dict.items():
        assert 0.0 <= prob <= 1.0, f"Probabilité hors [0,1] : {prob}"
    # La somme des top-3 doit être ≤ 1
    total_top3 = sum(top_dict.values())
    assert total_top3 <= 1.001, f"Somme top-3 = {total_top3}"
# =============================================================================
# TESTS EXPORTS (Partie 4.5)
# =============================================================================

def test_export_dataframe_creates_file(analytics):
    """Le helper _export_dataframe crée bien un fichier."""
    from pathlib import Path
    import app as app_module
    df = analytics.team_table
    if df.empty:
        pytest.skip("team_table vide")

    path = app_module._export_dataframe(df, prefix="test_export")
    assert path is not None
    assert Path(path).exists()
    # Nettoyage
    Path(path).unlink(missing_ok=True)


def test_export_dataframe_empty_returns_none():
    """Un DataFrame vide ne produit pas de fichier."""
    import pandas as pd
    import app as app_module
    empty_df = pd.DataFrame()
    path = app_module._export_dataframe(empty_df, prefix="test_empty")
    assert path is None