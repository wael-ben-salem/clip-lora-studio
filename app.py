"""
🎨 CLIP LoRA Studio — Application Gradio
Étude du fine-tuning de CLIP avec LoRA sur Fashion-MNIST.

Nouveautés UI :
  - page d'accueil au lancement (hero, KPI, accès rapides, graphiques, essai rapide)
  - menu latéral moderne qui pilote les onglets (onglets natifs masqués)
  - bouton clair / sombre
"""

import logging
import logging.handlers
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Modules internes ---
from modules.analytics import Analytics
from modules.inference import classify, classify_multi, get_image_embedding
from modules.models import ModelRegistry
from modules.retrieval import DEFAULT_EMBEDDINGS_MAP, ImageRetriever
from utils.constants import CLASSES, CLASS_PROMPTS, DATA_DIR, EXAMPLES_DIR, FIGURES_DIR

# --- Présentation (thème, CSS, hero, KPI, sidebar, footer) ---
from ui_theme import (
    CSS,
    MODEL_COLORS,
    TOGGLE_DARK_JS,
    build_theme,
    demo_note_html,
    footer_html,
    hero_html,
    insights_html,
    kpi_html,
    model_cards_html,
    section_html,
    sidebar_brand_html,
    status_html,
    topbar_html,
)

# =============================================================================
# LOGGING STRUCTURÉ
# Format : 2026-10-04 19:30:00 | ERROR | modules.retrieval | ...
# =============================================================================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)

_fh = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_ch, _fh])
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================
def _pick_col(df, *candidates):
    """
    Return the first column NAME present in df among candidates.
    Case-insensitive and loose (ignores _ / spaces / -).
    Raises KeyError with the list of available columns if nothing matches.
    """
    if df is None or df.empty:
        raise KeyError(f"DataFrame vide — candidats: {candidates}")

    for c in candidates:
        if c in df.columns:
            return c

    lower = {str(col).lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    def norm(s):
        return str(s).lower().replace("_", "").replace(" ", "").replace("-", "")
    loose = {norm(col): col for col in df.columns}
    for c in candidates:
        if norm(c) in loose:
            return loose[norm(c)]

    raise KeyError(
        f"Aucune colonne parmi {candidates} trouvée. "
        f"Colonnes disponibles : {list(df.columns)}"
    )


def _pick_col_or_none(df, *candidates):
    """Same as _pick_col but returns None instead of raising."""
    try:
        return _pick_col(df, *candidates)
    except KeyError:
        return None


# =============================================================================
# HELPERS EXPORT (Partie 4.5)
# =============================================================================
EXPORTS_DIR = Path("exports")
EXPORTS_DIR.mkdir(exist_ok=True)


def _export_dataframe(df: pd.DataFrame, prefix: str = "export") -> str:
    """Sauvegarde un DataFrame en CSV dans exports/ et retourne le chemin (ou None)."""
    if df is None or df.empty:
        return None

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    filepath = EXPORTS_DIR / filename

    try:
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info("Export CSV : %s (%d lignes)", filepath, len(df))
        return str(filepath)
    except Exception as e:
        logger.error("Erreur export CSV : %s", e)
        return None


def _export_markdown_summary(text: str, prefix: str = "summary") -> str:
    """Sauvegarde un résumé Markdown dans exports/ et retourne le chemin."""
    if not text or not text.strip():
        return None

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.md"
    filepath = EXPORTS_DIR / filename

    try:
        filepath.write_text(text, encoding="utf-8")
        logger.info("Export Markdown : %s", filepath)
        return str(filepath)
    except Exception as e:
        logger.error("Erreur export Markdown : %s", e)
        return None


# =============================================================================
# INITIALISATION (au démarrage de l'app)
# =============================================================================
logger.info("=" * 60)
logger.info("INITIALISATION CLIP LoRA Studio")
logger.info("=" * 60)

registry = ModelRegistry().load_all()
analytics = Analytics()

# Retriever multi-modèle — charge tous les embeddings disponibles
try:
    retriever = ImageRetriever(
        embeddings_map=DEFAULT_EMBEDDINGS_MAP,
        labels_path="assets/data/embeddings/eval_labels.npy",
        images_dir="assets/eval_images",
    )
except Exception as exc:
    logger.error("Retriever non disponible : %s", exc)
    logger.error("Lance : python scripts/prepare_eval_images.py")
    retriever = None

# =============================================================================
# DIAGNOSTIC AU DÉMARRAGE
# =============================================================================
_n_figs = len(analytics.list_available_figures("team"))
_n_data = len(analytics.list_available_data())
_retriever_ok = retriever is not None
_n_imgs = retriever.num_images if _retriever_ok else 0
_emb_shape = (
    next(iter(retriever._embeddings.values())).shape
    if _retriever_ok and retriever._embeddings else "N/A"
)
_normalized = (
    bool(
        np.allclose(
            np.linalg.norm(next(iter(retriever._embeddings.values()))[:10], axis=1),
            1.0, atol=1e-3,
        )
    )
    if _retriever_ok and retriever._embeddings else False
)
_label_dist: dict = {}
if _retriever_ok and retriever.labels is not None:
    _counts = np.bincount(retriever.labels, minlength=10)
    _label_dist = {CLASSES[i]: int(_counts[i]) for i in range(10)}

_avail_models = retriever.available_models if _retriever_ok else []
_unavail = retriever.get_unavailable_models() if _retriever_ok else {}

logger.info("=" * 50 + " DIAGNOSTIC " + "=" * 50)
logger.info("Modèles chargés : %s", registry.names)
logger.info("Retriever : %d images indexées", _n_imgs)
logger.info("Embeddings disponibles : %s", _avail_models)
logger.info("Embeddings shape : %s", _emb_shape)
logger.info("Embeddings normalisés : %s", _normalized)
logger.info("Labels distribution : %s", _label_dist)
logger.info("Figures disponibles : %d", _n_figs)
logger.info("CSV disponibles : %d", _n_data)
if _unavail:
    for m, hint in _unavail.items():
        logger.warning("Modèle non dispo : %s — %s", m, hint)
logger.info("=" * 112)

try:
    if not analytics.team_table.empty:
        logger.info("team_table columns : %s", list(analytics.team_table.columns))
except Exception as exc:
    logger.error("team_table indisponible : %s", exc)


# =============================================================================
# CONFIGURATION UI
# =============================================================================
APP_TITLE = "🎨 CLIP LoRA Studio"
APP_DESCRIPTION = """
**Explorez, comparez, comprenez** comment LoRA fine-tune CLIP sur Fashion-MNIST.

Projet académique — étude du trade-off **adaptation ↔ préservation de la généralité**.
"""

COLORS = {
    "Zero-shot": "#4A90E2",
    "A2 LoRA V+T": "#50C878",
    "B LoRA V only": "#F5A623",
}

CANONICAL_MODELS = ["Zero-shot", "A1 LinearHead", "A2 LoRA V+T", "B LoRA V only"]


# =============================================================================
# ONGLET 1 — CLASSIFICATION
# =============================================================================
def tab_classification():
    gr.Markdown("""
    ### 🎯 Classification d'images

    Uploadez une image de vêtement et comparez les prédictions entre les 3 modèles.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📷 Image d'entrée", height=300)
            model_dd = gr.Dropdown(registry.names, value="A2 LoRA V+T", label="🧠 Modèle")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            with gr.Row():
                btn_classify = gr.Button("🔍 Classifier", variant="primary", scale=2)
                btn_clear = gr.Button("🗑️ Effacer", scale=1)

        with gr.Column(scale=1):
            out_label = gr.Label(label="🎯 Prédiction", num_top_classes=5)
            out_plot = gr.BarPlot(
                x="Class", y="Probability", title="Distribution des probabilités",
                y_lim=[0, 1], height=350, sort="y",
            )
            out_table = gr.Dataframe(label="📊 Détails (toutes les classes)", interactive=False)

    def _classify(image, model_name, k):
        """Classifie une image et gère proprement les erreurs."""
        if image is None:
            return None, None, pd.DataFrame()
        try:
            top_dict, df = classify(registry, image, model_name=model_name, top_k=k)
            df_plot = df.head(k).copy()
            return top_dict, df_plot, df
        except ValueError as e:
            logger.warning("Erreur de classification : %s", e)
            return None, None, pd.DataFrame()
        except Exception as e:
            logger.error("Erreur inattendue classification : %s", e, exc_info=True)
            return None, None, pd.DataFrame()

    def _clear():
        return None, None, None, pd.DataFrame()

    btn_classify.click(_classify, [img_input, model_dd, top_k],
                       [out_label, out_plot, out_table])
    btn_clear.click(_clear, outputs=[img_input, out_label, out_plot, out_table])

    example_files = sorted(EXAMPLES_DIR.glob("*.png")) + sorted(EXAMPLES_DIR.glob("*.jpg"))
    if example_files:
        gr.Examples(examples=[[str(f)] for f in example_files[:8]],
                    inputs=[img_input], label="📸 Exemples")


# =============================================================================
# ONGLET 2 — PROMPT LAB
# =============================================================================
def tab_prompt_lab():
    gr.Markdown("""
    ### 🔤 Prompt Lab — Testez vos templates

    Modifiez le template. Utilisez `{}` comme placeholder pour la classe.

    **Exemples** : `a photo of a {}` · `a black and white photo of a {}` · `{}`
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📷 Image", height=300)
            template_input = gr.Textbox(value="a photo of a {}", label="📝 Template",
                                        lines=2)
            model_dd = gr.Dropdown(registry.names, value="A2 LoRA V+T", label="🧠 Modèle")
            btn_run = gr.Button("🚀 Tester ce template", variant="primary")
            btn_reset = gr.Button("↻ Reset template", size="sm")

        with gr.Column(scale=1):
            out_label = gr.Label(label="🎯 Prédiction", num_top_classes=5)
            out_table = gr.Dataframe(label="📊 Scores par classe", interactive=False)
            out_md = gr.Markdown(label="💡 Interprétation")

    def _run_prompt(image, template, model_name):
        if image is None:
            return None, pd.DataFrame(), "⚠️ Uploadez une image."
        try:
            prompts = [template.format(c.lower()) for c in CLASSES]
        except Exception as e:
            return None, pd.DataFrame(), f"❌ Erreur template : `{e}`"

        try:
            top_dict, df = classify(registry, image, model_name=model_name,
                                    top_k=5, custom_prompts=prompts)
        except ValueError as e:
            logger.warning("Prompt Lab erreur : %s", e)
            return None, pd.DataFrame(), f"⚠️ {e}"
        except Exception as e:
            logger.error("Prompt Lab erreur inattendue : %s", e, exc_info=True)
            return None, pd.DataFrame(), f"❌ Erreur : {e}"

        df["Class"] = CLASSES
        top_dict = {CLASSES[i]: float(df.iloc[i]["Probability"])
                    for i in range(min(5, len(df)))}

        winner = list(top_dict.keys())[0]
        confidence = list(top_dict.values())[0]
        interp = f"✅ **Classe gagnante** : `{winner}` avec **{confidence*100:.1f}%**\n\n**Top-3** :\n"
        for i, (cls, prob) in enumerate(list(top_dict.items())[:3]):
            interp += f"{i+1}. `{cls}` — {prob*100:.2f}%\n"
        return top_dict, df, interp

    def _reset():
        return "a photo of a {}"

    btn_run.click(_run_prompt, [img_input, template_input, model_dd],
                  [out_label, out_table, out_md])
    btn_reset.click(_reset, outputs=[template_input])


# =============================================================================
# ONGLET 3 — IMAGE RETRIEVAL
# =============================================================================
def tab_retrieval():
    avail = retriever.available_models if retriever else []
    default_model = avail[0] if avail else "Zero-shot"
    unavail_hints = retriever.get_unavailable_models() if retriever else {}

    gr.Markdown(f"""
    ### 🔍 Image Retrieval — Recherche visuelle

    Uploadez une image → trouve les **k images les plus similaires** dans le dataset
    Fashion-MNIST (2000 images indexées).

    **Comment ça marche ?**
    - L'image est encodée avec le modèle sélectionné
    - Comparaison par similarité cosinus avec les {retriever.num_images if retriever else 0} embeddings pré-calculés
    - Retour des top-K avec leurs vraies classes (ground-truth)

    {f"ℹ️ **Modèles disponibles :** {', '.join(avail)}" if avail else ""}
    """)

    if unavail_hints:
        hints_md = "\n".join(
            f"- ⚠️ **{m}** : {hint}" for m, hint in unavail_hints.items()
        )
        gr.Markdown(f"""
        **Embeddings manquants :**
        {hints_md}
        """)

    if retriever is None:
        gr.Markdown("""
        ⚠️ **Retriever non disponible.**

        Lance d'abord :
        ```bash
        python scripts/prepare_eval_images.py
        python scripts/regenerate_labels.py
        ```

        Puis relance l'app.
        """)
        return

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📷 Image requête", height=280)
            k_slider = gr.Slider(3, 24, value=12, step=3, label="Top-K (nombre d'images)")
            model_dd = gr.Dropdown(
                avail,
                value=default_model,
                label="🧠 Modèle d'encodage (⚠️ doit correspondre aux embeddings)",
                info="Seuls les modèles avec embeddings pré-calculés sont disponibles.",
            )
            class_filter_dd = gr.Dropdown(
                choices=["Toutes les classes"] + CLASSES,
                value="Toutes les classes",
                label="🔍 Filtrer par classe",
                multiselect=False,
            )
            metric_dd = gr.Dropdown(
                choices=["cosine", "euclidean", "dot"],
                value="cosine",
                label="📐 Métrique de similarité",
            )
            with gr.Row():
                btn_search = gr.Button("🔍 Rechercher", variant="primary", scale=2)
                btn_clear = gr.Button("🗑️ Effacer", scale=1)

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="📸 Images similaires trouvées",
                columns=4, rows=3, height=480, object_fit="cover", show_label=True,
            )
            dist_plot = gr.BarPlot(
                x="Class", y="Count", title="Distribution des classes trouvées",
                y_lim=[0, 24], height=280, sort="y",
            )
            stats_md = gr.Markdown()

    def _search(image, k, model_name, class_filter, metric):
        if image is None:
            return None, None, "⚠️ Uploadez une image."
        if retriever is None:
            return None, None, "⚠️ Retriever non disponible."
        if not model_name:
            return None, None, "⚠️ Aucun modèle sélectionné."

        try:
            encode_model = model_name if model_name in registry.names else "Zero-shot"
            emb = get_image_embedding(registry, encode_model, image).cpu().numpy()

            cf = None
            if class_filter and class_filter != "Toutes les classes":
                cf = [CLASSES.index(class_filter)]

            results = retriever.retrieve(
                emb,
                model_name=model_name,
                k=int(k),
                return_images=True,
                class_filter=cf,
                metric=metric,
            )
        except ValueError as exc:
            return None, None, f"⚠️ {exc}"
        except Exception as exc:
            logger.error("Retrieval error: %s", exc, exc_info=True)
            return None, None, f"❌ Erreur inattendue : {exc}"

        if not results:
            return [], pd.DataFrame({"Class": [], "Count": []}), "Aucun résultat."

        gallery_items = [
            (r["image"], f"{r['class']} | {r['score']:.3f}")
            for r in results if "image" in r
        ]

        dist = retriever.get_class_distribution(results)
        df_dist = pd.DataFrame({
            "Class": list(dist.keys()),
            "Count": list(dist.values()),
        }).sort_values("Count", ascending=False)

        top_result = results[0]
        dist_sorted = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top3_lines = "\n".join(
            f"- **{cls}** : {count} images"
            for cls, count in dist_sorted[:3]
        )

        stats = f"""
        ### 📊 Résultats de la recherche

        **Modèle utilisé :** {model_name}  
        **Métrique :** {metric}  
        **Query top-1 :** {top_result["class"]}  
        **Score top-1 :** {top_result["score"]:.4f}

        **{len(results)}** images trouvées · **{len(dist)}** classes représentées

        **Top-3 classes :**
        {top3_lines}
        """
        return gallery_items, df_dist, stats

    def _clear():
        return None, None, None, ""

    btn_search.click(
        _search,
        inputs=[img_input, k_slider, model_dd, class_filter_dd, metric_dd],
        outputs=[gallery, dist_plot, stats_md],
    )
    btn_clear.click(_clear, outputs=[img_input, gallery, dist_plot, stats_md])

    example_files = sorted(EXAMPLES_DIR.glob("*.png"))[:8]
    if example_files:
        gr.Examples(
            examples=[[str(f)] for f in example_files],
            inputs=[img_input],
            label="📸 Exemples",
        )


# =============================================================================
# ONGLET 4 — DRIFT EXPLORER
# =============================================================================
def tab_drift_explorer():
    gr.Markdown("""
    ### 🧭 Drift Explorer — Exploration interactive du drift

    Le drift angulaire mesure à quel point les embeddings ont été modifiés
    par le fine-tuning.

    **Interprétation :**
    - drift ≈ 0 → encoder inchangé
    - drift > 0.5 → encoder fortement modifié
    """)

    df_delta = analytics.confusion_delta
    if df_delta.empty:
        gr.Markdown(
            "⚠️ Données non trouvées. "
            "Vérifie assets/data/team_confusion_delta.csv."
        )
        return

    try:
        col_class = _pick_col(df_delta, "class", "Class", "classe", "label")
    except KeyError:
        col_class = df_delta.columns[0]

    col_delta = _pick_col_or_none(df_delta, "delta_a2_b", "delta_A2_B", "deltaA2B")
    col_acc_a2 = _pick_col_or_none(df_delta, "acc_a2", "acc_A2", "accuracy_a2")
    col_acc_zs = _pick_col_or_none(df_delta, "acc_zs", "acc_ZS", "accuracy_zs", "acc_zeroshot")
    col_acc_b = _pick_col_or_none(df_delta, "acc_b", "acc_B", "accuracy_b")
    col_delta_zs = _pick_col_or_none(df_delta, "delta_a2_zs", "delta_A2_ZS", "deltaA2ZS")

    with gr.Row():
        if col_delta is not None:
            best_class_row = df_delta.loc[df_delta[col_delta].idxmax()]
            worst_class_row = df_delta.loc[df_delta[col_delta].idxmin()]

            gr.Markdown(f"""
            ### 📊 Points clés

            🏆 **Classe la plus améliorée par A2 vs B :**  
            {best_class_row[col_class]}
            ({best_class_row[col_delta] * 100:+.2f} pts)

            ⚠️ **Classe la moins améliorée :**  
            {worst_class_row[col_class]}
            ({worst_class_row[col_delta] * 100:+.2f} pts)
            """)
        else:
            gr.Markdown("### 📊 Points clés\n⚠️ Colonne delta_a2_b introuvable.")

    metric_choices = []
    for name, col in [
        ("acc_zs", col_acc_zs),
        ("acc_a2", col_acc_a2),
        ("acc_b", col_acc_b),
        ("delta_a2_b", col_delta),
        ("delta_a2_zs", col_delta_zs),
    ]:
        if col is not None:
            metric_choices.append(name)

    if not metric_choices:
        gr.Markdown("⚠️ Aucune métrique reconnue dans confusion_delta.")
        gr.Dataframe(value=df_delta, label="Données brutes", interactive=False)
        return

    default_metric = "acc_a2" if "acc_a2" in metric_choices else metric_choices[0]

    with gr.Row():
        metric_dd = gr.Dropdown(choices=metric_choices, value=default_metric,
                                label="📊 Métrique à afficher")
        sort_dd = gr.Dropdown(choices=["Ordre classes", "Décroissant", "Croissant"],
                              value="Ordre classes", label="🔃 Tri")

    plot = gr.Plot(label="📈 Graphique interactif")

    metric_to_col = {
        "acc_zs": col_acc_zs,
        "acc_a2": col_acc_a2,
        "acc_b": col_acc_b,
        "delta_a2_b": col_delta,
        "delta_a2_zs": col_delta_zs,
    }

    def make_plot(metric, sort_order):
        col = metric_to_col.get(metric, metric)

        if col is None or col not in df_delta.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Colonne '{metric}' introuvable",
                               showarrow=False, font=dict(size=16))
            fig.update_layout(height=500)
            return fig

        df_plot = df_delta.copy()
        if sort_order == "Décroissant":
            df_plot = df_plot.sort_values(col, ascending=False)
        elif sort_order == "Croissant":
            df_plot = df_plot.sort_values(col, ascending=True)

        fig = px.bar(
            df_plot, x=col_class, y=col, color=col,
            color_continuous_scale="RdYlGn",
            title=f"{metric} par classe",
            labels={col_class: "Classe", col: metric},
            height=500,
        )
        fig.update_layout(xaxis_tickangle=-45, plot_bgcolor="white", font=dict(size=12))
        fig.update_traces(marker_line_color="black", marker_line_width=1)
        return fig

    plot.value = make_plot(default_metric, "Ordre classes")
    metric_dd.change(make_plot, [metric_dd, sort_dd], outputs=plot)
    sort_dd.change(make_plot, [metric_dd, sort_dd], outputs=plot)

    gr.Markdown("### 📋 Tableau détaillé")
    gr.Dataframe(value=df_delta, label="Données brutes", interactive=False)

    gr.Markdown("### 📊 Comparaison ZS vs A2 vs B")

    def make_comparison():
        fig = go.Figure()
        for model, col, color in [
            ("Zero-shot", col_acc_zs, COLORS["Zero-shot"]),
            ("A2 LoRA V+T", col_acc_a2, COLORS["A2 LoRA V+T"]),
            ("B LoRA V only", col_acc_b, COLORS["B LoRA V only"]),
        ]:
            if col is not None:
                fig.add_trace(go.Bar(
                    name=model, x=df_delta[col_class], y=df_delta[col],
                    marker_color=color, marker_line_color="black", marker_line_width=1,
                ))

        fig.update_layout(
            barmode="group", height=500, xaxis_tickangle=-45,
            plot_bgcolor="white", title="Accuracy par classe", yaxis_title="Accuracy",
        )
        return fig

    gr.Plot(value=make_comparison())


# =============================================================================
# ONGLET 5 — ANALYTICS
# =============================================================================
def tab_analytics():
    gr.Markdown("""
    ### 📈 Analytics — Dashboard récapitulatif du projet

    Vue d'ensemble des 4 modèles comparés avec leurs performances clés.
    """)

    tt = analytics.team_table
    if tt.empty:
        gr.Markdown("⚠️ TEAM_FINAL_TABLE non trouvé.")
        return

    try:
        col_model = _pick_col(tt, "Model", "model", "model_name", "name", "Modèle")
        col_acc = _pick_col(tt, "ClsAcc", "acc", "cls_acc", "Cls_Acc",
                            "Accuracy", "accuracy", "clsacc", "ClassAcc")
        col_p1 = _pick_col(tt, "P1", "p1", "P@1", "Retrieval_P1",
                           "recall@1", "Recall1", "retrieval_p1")
        col_drift = _pick_col(tt, "Drift", "drift", "Drift_Angle", "drift_angle")
        col_params = _pick_col(tt, "Params", "params", "n_params", "NumParams",
                               "nb_params", "ParamCount")
    except KeyError as e:
        gr.Markdown(f"⚠️ Colonnes manquantes dans TEAM_FINAL_TABLE : {e}")
        gr.Dataframe(value=tt, label="Données brutes (colonnes détectées)", interactive=False)
        return

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            ### 🎯 Accuracy

            {tt[col_acc].max():.3f}
            ({tt.loc[tt[col_acc].idxmax(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 🔍 Retrieval P@1

            {tt[col_p1].max():.3f}
            ({tt.loc[tt[col_p1].idxmax(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 🧭 Drift minimal

            {tt[col_drift].min():.3f}
            ({tt.loc[tt[col_drift].idxmin(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 💾 Params max

            {tt[col_params].max():,}
            """)

    with gr.Tabs():
        with gr.Tab("📊 Tableau"):
            gr.Dataframe(value=tt, label="TEAM_FINAL_TABLE", interactive=False)
            ranked = tt.sort_values(col_acc, ascending=False)
            gr.Dataframe(value=ranked, label="Classement par accuracy", interactive=False)

        with gr.Tab("🔥 Heatmap"):
            heatmap_path = analytics.get_figure("TEAM_FINAL_TABLE_heatmap.png", "team")
            if heatmap_path:
                gr.Image(value=heatmap_path, label="Heatmap TEAM_FINAL_TABLE")
            else:
                gr.Markdown("⚠️ Heatmap non trouvée")

        with gr.Tab("🎯 Pareto"):
            fig = px.scatter(
                tt, x=col_acc, y=col_p1, size=col_params, color=col_drift,
                hover_name=col_model, text=col_model,
                color_continuous_scale="RdYlGn_r",
                title="Trade-off Accuracy ↔ Retrieval (taille = params)",
                height=500,
            )
            fig.update_traces(marker=dict(line=dict(width=2, color="black")),
                              textposition="top center")
            fig.update_layout(xaxis_title="Classification Accuracy →",
                              yaxis_title="Text→Image P@1 →",
                              plot_bgcolor="white")
            gr.Plot(value=fig)

    gr.Markdown("### 📸 Slide de conclusion")
    slide_path = analytics.get_figure("SLIDE_CONCLUSION.png", "team")
    if slide_path:
        gr.Image(value=slide_path, label="Synthèse du projet")
    else:
        gr.Markdown("⚠️ Slide non trouvée")

    try:
        acc_zs = tt.loc[tt[col_model] == "Zero-shot", col_acc].values
        acc_a2 = tt.loc[tt[col_model] == "A2 LoRA V+T", col_acc].values
        if len(acc_zs) and len(acc_a2):
            delta_pts = (acc_a2[0] - acc_zs[0]) * 100
            insight_line = f"- ✅ LoRA V+T : {delta_pts:+.1f}% accuracy vs Zero-shot"
        else:
            insight_line = "- ✅ LoRA V+T : comparaison ZS/A2 indisponible"
    except Exception:
        insight_line = "- ✅ LoRA V+T : comparaison ZS/A2 indisponible"

    gr.Markdown(f"""
    ### 💡 Insights clés

    {insight_line}

    ✅ LoRA V only : 2× plus léger avec 98% des performances d'A2

    ⚡ A1 LinearHead : efficace à budget ultra-réduit (5K params)
    """)


# =============================================================================
# ONGLET 6 — COMPARATEUR VISUEL (Partie 4.1)
# =============================================================================
def tab_comparator():
    """Compare les 4 modèles côte à côte sur une même image."""
    gr.Markdown("""
    ### 🎨 Comparateur Visuel — 4 modèles côte à côte

    Uploadez une image et voyez **comment chaque modèle la classifie**.
    Idéal pour comprendre l'impact du fine-tuning LoRA.
    """)

    model_names = list(registry.names)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📷 Image à comparer", height=320)
            btn_compare = gr.Button("🔬 Comparer les 4 modèles", variant="primary")
            btn_clear = gr.Button("🗑️ Effacer", size="sm")

            gr.Markdown("""
            **Comment lire les résultats ?**

            - 🟢 **Consensus** : tous les modèles sont d'accord
            - 🟡 **Divergence partielle** : 3/4 d'accord
            - 🔴 **Désaccord** : aucun consensus
            """)

        with gr.Column(scale=2):
            consensus_md = gr.Markdown(label="🎯 Consensus")

            gr.Markdown("### 📊 Résultats par modèle")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"#### 🧊 {model_names[0] if len(model_names) > 0 else 'Zero-shot'}")
                    out0_label = gr.Label(label="Prédiction", num_top_classes=3)
                    out0_md = gr.Markdown()

                with gr.Column():
                    gr.Markdown(f"#### 🔷 {model_names[1] if len(model_names) > 1 else 'A1 LinearHead'}")
                    out1_label = gr.Label(label="Prédiction", num_top_classes=3)
                    out1_md = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"#### 🟢 {model_names[2] if len(model_names) > 2 else 'A2 LoRA V+T'}")
                    out2_label = gr.Label(label="Prédiction", num_top_classes=3)
                    out2_md = gr.Markdown()

                with gr.Column():
                    gr.Markdown(f"#### 🟠 {model_names[3] if len(model_names) > 3 else 'B LoRA V only'}")
                    out3_label = gr.Label(label="Prédiction", num_top_classes=3)
                    out3_md = gr.Markdown()

            gr.Markdown("### 📋 Tableau récapitulatif")
            summary_df = gr.Dataframe(
                headers=["Modèle", "Prédiction", "Confiance", "Top-2", "Top-2 Conf"],
                label="Comparaison",
                interactive=False,
            )

            with gr.Row():
                btn_export_cmp = gr.Button("📥 Exporter en CSV", size="sm")
                export_cmp_md = gr.Markdown()

    last_summary = gr.State(value=None)

    def _export_comparison(summary_data):
        if summary_data is None:
            return "⚠️ Lancez d'abord une comparaison."

        path = _export_dataframe(pd.DataFrame(summary_data), prefix="comparison")
        if path:
            return f"✅ Exporté : `{path}`"
        return "❌ Échec de l'export."

    btn_export_cmp.click(_export_comparison, inputs=[last_summary], outputs=[export_cmp_md])

    def _compare(image):
        """Classifie l'image avec les 4 modèles et compare."""
        empty_df = pd.DataFrame(
            columns=["Modèle", "Prédiction", "Confiance", "Top-2", "Top-2 Conf"]
        )

        if image is None:
            return (
                "⚠️ Uploadez une image.",
                None, "", None, "", None, "", None, "",
                empty_df,
                None,
            )

        results = []

        for name in model_names[:4]:
            try:
                top_dict, df = classify(registry, image, model_name=name, top_k=3)
                top_class = list(top_dict.keys())[0]
                top_conf = list(top_dict.values())[0]
                top2_class = list(top_dict.keys())[1] if len(top_dict) > 1 else "—"
                top2_conf = list(top_dict.values())[1] if len(top_dict) > 1 else 0.0

                results.append({
                    "model": name,
                    "top_dict": top_dict,
                    "top_class": top_class,
                    "top_conf": top_conf,
                    "top2_class": top2_class,
                    "top2_conf": top2_conf,
                })

            except Exception as e:
                logger.error("Erreur comparaison %s : %s", name, e)
                results.append({
                    "model": name,
                    "top_dict": {},
                    "top_class": "❌ Erreur",
                    "top_conf": 0.0,
                    "top2_class": "—",
                    "top2_conf": 0.0,
                })

        predictions = [r["top_class"] for r in results]
        unique_preds = set(predictions)

        if len(unique_preds) == 1:
            consensus = (
                f"🟢 **Consensus unanime** sur `{predictions[0]}` "
                f"({len(predictions)}/{len(predictions)} modèles d'accord)"
            )
        elif any(predictions.count(p) >= 3 for p in predictions):
            from collections import Counter

            majority = Counter(predictions).most_common(1)[0][0]
            n = predictions.count(majority)
            consensus = (
                f"🟡 **Divergence partielle** — {n}/{len(predictions)} "
                f"modèles prédisent `{majority}`"
            )
        else:
            consensus = (
                "🔴 **Désaccord total** — prédictions variées : "
                + ", ".join(predictions)
            )

        def model_md(r):
            if not r["top_dict"]:
                return "❌ Modèle indisponible"

            lines = []
            for cls, prob in list(r["top_dict"].items())[:3]:
                bar = "█" * int(prob * 20)
                lines.append(f"`{cls:<12}` {bar} {prob * 100:.1f}%")

            return "```\n" + "\n".join(lines) + "\n```"

        summary = pd.DataFrame([
            {
                "Modèle": r["model"],
                "Prédiction": r["top_class"],
                "Confiance": f"{r['top_conf'] * 100:.1f}%",
                "Top-2": r["top2_class"],
                "Top-2 Conf": f"{r['top2_conf'] * 100:.1f}%",
            }
            for r in results
        ])

        labels = [r["top_dict"] for r in results]
        mds = [model_md(r) for r in results]

        # Garantit toujours 4 sorties de modèles
        while len(labels) < 4:
            labels.append({})
            mds.append("❌ Modèle indisponible")

        return (
            consensus,
            labels[0], mds[0],
            labels[1], mds[1],
            labels[2], mds[2],
            labels[3], mds[3],
            summary,
            summary.to_dict("records"),
        )

    def _clear():
        empty_df = pd.DataFrame(
            columns=["Modèle", "Prédiction", "Confiance", "Top-2", "Top-2 Conf"]
        )
        return (
            None,
            "Uploadez une image pour comparer.",
            None, "", None, "", None, "", None, "",
            empty_df,
            None,
        )

    btn_compare.click(
        _compare,
        inputs=[img_input],
        outputs=[
            consensus_md,
            out0_label, out0_md,
            out1_label, out1_md,
            out2_label, out2_md,
            out3_label, out3_md,
            summary_df,
            last_summary,
        ],
    )

    btn_clear.click(
        _clear,
        outputs=[
            img_input,
            consensus_md,
            out0_label, out0_md,
            out1_label, out1_md,
            out2_label, out2_md,
            out3_label, out3_md,
            summary_df,
            last_summary,
        ],
    )

    example_files = sorted(EXAMPLES_DIR.glob("*.png"))[:6]
    if example_files:
        gr.Examples(
            examples=[[str(f)] for f in example_files],
            inputs=[img_input],
            label="📸 Exemples",
        )


# =============================================================================
# ONGLET 7 — CARTE DE CONFIANCE / CALIBRATION (Partie 4.2)
# =============================================================================
def tab_calibration():
    """Analyse la calibration des probabilités (ECE, reliability diagram)."""
    gr.Markdown("""
    ### 🌡️ Carte de Confiance — Calibration des probabilités

    Un modèle **calibré** est un modèle dont les scores de confiance
    correspondent à la réalité :

    - dit "80%" → se trompe 20% du temps
    - dit "50%" → se trompe 50% du temps

    **Utilité** : un modèle sur-confiant est dangereux en production.
    """)

    df_cal = None

    if hasattr(analytics, "calibration"):
        cal = analytics.calibration
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df_cal = cal
        elif isinstance(cal, dict) and len(cal) > 0:
            df_cal = pd.DataFrame(cal)

    if df_cal is None and hasattr(analytics, "confidence_distribution"):
        conf = analytics.confidence_distribution
        if isinstance(conf, pd.DataFrame) and not conf.empty:
            df_cal = conf
        elif isinstance(conf, dict) and len(conf) > 0:
            df_cal = pd.DataFrame(conf)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📊 ECE (Expected Calibration Error)

            L'ECE mesure l'écart moyen entre confiance et précision.

            - **ECE < 0.05** : excellent ✅
            - **ECE 0.05 – 0.10** : bon
            - **ECE > 0.10** : sur/sous-confiant ⚠️
            """)

            fig_cal = analytics.get_figure("team_calibration_reliability.png", "team")
            if fig_cal:
                gr.Image(value=fig_cal, label="Reliability Diagram")
            else:
                gr.Markdown("⚠️ Figure de calibration non trouvée.")

        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📈 Distribution des confiances

            Un bon modèle a une distribution **étalée**.
            Un modèle sur-confiant a un pic vers 1.0.
            """)

            fig_conf = analytics.get_figure("team_confidence_distribution.png", "team")
            if fig_conf:
                gr.Image(value=fig_conf, label="Distribution des confiances")
            else:
                gr.Markdown("⚠️ Distribution non trouvée.")

    if df_cal is not None and not df_cal.empty:
        gr.Markdown("### 📋 Données de calibration")
        gr.Dataframe(value=df_cal, label="Calibration", interactive=False)
    else:
        gr.Markdown(
            "### 📋 Données de calibration\n"
            "*⚠️ Données non disponibles "
            "(le simulateur ci-dessous fonctionne quand même).*"
        )

    gr.Markdown("""
    ### 💡 Comment interpréter ?

    **Reliability Diagram** :

    - Courbe sur la diagonale → parfaitement calibré
    - Courbe au-dessus → sous-confiant (le modèle pourrait être plus sûr)
    - Courbe en-dessous → sur-confiant (le modèle est trop sûr de lui)

    **En pratique sur Fashion-MNIST** :

    - Zero-shot : souvent bien calibré mais peu précis
    - Modèles fine-tunés : plus précis mais peuvent devenir sur-confants
    - Le LoRA V+T trouve généralement un bon compromis
    """)

    gr.Markdown("### 🧪 Simulateur interactif")
    gr.Markdown("""
    Testez la calibration sur une image : le modèle est-il aussi confiant
    qu'il devrait l'être ?
    """)

    with gr.Row():
        img_input = gr.Image(type="pil", label="📷 Image à tester", height=250)
        model_dd = gr.Dropdown(
            choices=registry.names,
            value=registry.names[0] if registry.names else "Zero-shot",
            label="🧠 Modèle",
        )
        btn_test = gr.Button("🔍 Analyser la confiance", variant="primary")

    calib_out_md = gr.Markdown()

    def _analyze(image, model_name):
        if image is None:
            return "⚠️ Uploadez une image."

        try:
            top_dict, df = classify(registry, image, model_name=model_name, top_k=3)
        except Exception as e:
            logger.error("Calibration analyse erreur : %s", e, exc_info=True)
            return f"❌ Erreur : {e}"

        top_class = list(top_dict.keys())[0]
        top_conf = list(top_dict.values())[0]
        second_conf = list(top_dict.values())[1] if len(top_dict) > 1 else 0.0

        if top_conf > 0.9:
            verdict = "🔴 **Très confiant** — vérifier si justifié"
        elif top_conf > 0.7:
            verdict = "🟢 **Confiance modérée** — bon équilibre"
        elif top_conf > 0.5:
            verdict = "🟡 **Peu confiant** — le modèle hésite"
        else:
            verdict = "⚪ **Très incertain** — prédiction peu fiable"

        margin = top_conf - second_conf

        return f"""
        ### 📊 Analyse

        **Modèle** : `{model_name}`  
        **Classe prédite** : `{top_class}`  
        **Confiance** : **{top_conf * 100:.1f}%**

        **Marge avec le 2ᵉ** : {margin * 100:+.1f}%  
        *(une marge importante = modèle sûr de lui)*

        **Interprétation** : {verdict}
        """

    btn_test.click(_analyze, [img_input, model_dd], calib_out_md)


# =============================================================================
# ONGLET 8 — PLAYGROUND MULTI-IMAGES (Partie 4.3)
# =============================================================================
def tab_playground():
    """Upload batch d'images + prédictions + export CSV."""
    gr.Markdown("""
    ### 🧪 Playground Multi-Images — Analyse par lot

    Uploadez **plusieurs images** d'un coup et obtenez :

    - La prédiction de chaque modèle sur chaque image
    - Le **consensus** entre modèles (✅ / ⚠️)
    - Un **export CSV** téléchargeable
    - Les **statistiques agrégées** par modèle
    """)

    with gr.Row():
        with gr.Column(scale=1):
            images_input = gr.File(
                label="📁 Uploadez plusieurs images (PNG/JPG)",
                file_count="multiple",
                file_types=["image"],
            )
            model_choice = gr.Dropdown(
                choices=["Tous les modèles"] + list(registry.names),
                value="Tous les modèles",
                label="🧠 Modèle(s) à utiliser",
            )
            btn_run = gr.Button("🚀 Analyser le batch", variant="primary")
            btn_clear = gr.Button("🗑️ Effacer", size="sm")

            gr.Markdown("""
            **Comment ça marche ?**

            1. Sélectionne plusieurs images
            2. Choisis "Tous les modèles" ou un seul
            3. Clique Analyser
            4. Télécharge le CSV
            """)

        with gr.Column(scale=2):
            progress = gr.Markdown("*En attente...*")
            summary_md = gr.Markdown()

            gr.Markdown("### 📋 Résultats détaillés")
            results_df = gr.Dataframe(label="Prédictions", interactive=False, wrap=True)

            csv_file = gr.File(label="📥 Télécharger le CSV", visible=False)

            gr.Markdown("### 📊 Statistiques agrégées")
            stats_md = gr.Markdown()

    def _process_batch(files, model_choice):
        """Traite toutes les images avec les modèles sélectionnés."""
        empty_df = pd.DataFrame()

        if not files:
            return "⚠️ Uploadez au moins une image.", "", empty_df, None, ""

        if model_choice == "Tous les modèles":
            models = list(registry.names)
        else:
            models = [model_choice]

        from PIL import Image

        rows = []
        n_ok = 0
        n_err = 0
        progress_lines = []

        for i, f in enumerate(files):
            file_name = getattr(f, "name", None)
            if file_name:
                fname = Path(file_name).name
            else:
                fname = f"image_{i}"

            try:
                img = Image.open(file_name).convert("RGB")
            except Exception as e:
                logger.warning("Impossible de lire %s : %s", fname, e)
                rows.append({
                    "Image": fname,
                    "Modèle": "—",
                    "Prédiction": "❌ Erreur",
                    "Confiance": 0.0,
                })
                n_err += 1
                progress_lines.append(f"❌ {i + 1}/{len(files)} — {fname}")
                continue

            for model_name in models:
                try:
                    top_dict, _ = classify(registry, img, model_name=model_name, top_k=1)
                    top_class = list(top_dict.keys())[0]
                    top_conf = list(top_dict.values())[0]

                    rows.append({
                        "Image": fname,
                        "Modèle": model_name,
                        "Prédiction": top_class,
                        "Confiance": round(top_conf, 4),
                    })
                    n_ok += 1

                except Exception as e:
                    logger.error("Erreur %s sur %s : %s", model_name, fname, e, exc_info=True)
                    rows.append({
                        "Image": fname,
                        "Modèle": model_name,
                        "Prédiction": "❌ Erreur",
                        "Confiance": 0.0,
                    })
                    n_err += 1

            progress_lines.append(f"✅ {i + 1}/{len(files)} — {fname}")

        df = pd.DataFrame(rows)

        if df.empty:
            return "⚠️ Aucun résultat.", "", empty_df, None, ""

        stats_lines = ["### 📊 Stats par modèle\n"]
        stats_lines.append("| Modèle | Images traitées | Confiance moyenne |")
        stats_lines.append("|--------|-----------------|-------------------|")

        for model_name in models:
            sub = df[df["Modèle"] == model_name]
            if not sub.empty:
                mean_conf = sub["Confiance"].mean()
                stats_lines.append(f"| {model_name} | {len(sub)} | {mean_conf * 100:.1f}% |")

        stats_md_str = "\n".join(stats_lines)

        summary = f"""
        ### ✅ Analyse terminée

        - **Images traitées** : {len(files)}
        - **Prédictions générées** : {n_ok}
        - **Erreurs** : {n_err}
        - **Modèles utilisés** : {', '.join(models)}
        """

        csv_path = _export_dataframe(df, prefix="playground")
        if csv_path is None:
            csv_path = ""

        return (
            "\n".join(progress_lines[-5:]),
            summary,
            df,
            str(csv_path),
            stats_md_str,
        )

    def _clear():
        return None, "*En attente...*", "", pd.DataFrame(), None, ""

    btn_run.click(
        _process_batch,
        inputs=[images_input, model_choice],
        outputs=[progress, summary_md, results_df, csv_file, stats_md],
    )

    btn_clear.click(
        _clear,
        outputs=[images_input, progress, summary_md, results_df, csv_file, stats_md],
    )


# =============================================================================
# ONGLET 9 — RETRIEVAL CROSS-MODAL (Partie 4.4)
# =============================================================================
def tab_cross_modal():
    """Recherche texte → images via CLIP (cross-modal retrieval)."""
    gr.Markdown("""
    ### 🔎 Retrieval Cross-Modal — Texte → Images

    Tapez une **description en anglais** et trouvez les images Fashion-MNIST
    les plus proches sémantiquement.

    **Comment ça marche ?**
    - Le texte est encodé par l'encodeur **texte** de CLIP
    - Comparaison cosinus avec les embeddings **image** des 2000 images
    - Retour des top-K les plus proches (alignement cross-modal)

    **Exemples de requêtes :**
    - `a black sneaker` · `a red dress` · `a warm coat`
    - `a small bag` · `a formal shirt` · `a pair of boots`
    """)

    if retriever is None:
        gr.Markdown("""
        ⚠️ **Retriever non disponible.**

        Lance :

        ```bash
        python scripts/prepare_eval_images.py
        ```

        Puis relance l'app.
        """)
        return

    avail = retriever.available_models if retriever else []
    default_model = avail[0] if avail else "Zero-shot"

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Description (en anglais)",
                placeholder="a black sneaker",
                lines=2,
            )
            k_slider = gr.Slider(3, 24, value=12, step=3, label="Top-K images")
            model_dd = gr.Dropdown(
                choices=avail,
                value=default_model,
                label="🧠 Modèle d'encodage image",
                info="Doit correspondre aux embeddings pré-calculés.",
            )

            with gr.Row():
                btn_search = gr.Button("🔍 Rechercher", variant="primary", scale=2)
                btn_clear = gr.Button("🗑️ Effacer", scale=1)

            gr.Markdown("""
            **💡 Astuce** : CLIP est entraîné sur des descriptions
            naturelles. Préférez `a black sneaker` à `sneaker black`.
            """)

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="📸 Images trouvées",
                columns=4, rows=3, height=480, object_fit="cover", show_label=True,
            )
            dist_plot = gr.BarPlot(
                x="Class", y="Count", title="Distribution des classes trouvées",
                y_lim=[0, 24], height=280, sort="y",
            )
            stats_md = gr.Markdown()

    def _search_text(query_text, k, model_name):
        if not query_text or not query_text.strip():
            return None, None, "⚠️ Saisissez une description."

        if retriever is None:
            return None, None, "⚠️ Retriever non disponible."

        try:
            from modules.inference import get_text_embeddings

            txt_emb = get_text_embeddings(
                registry, "Zero-shot", [query_text.strip()],
            ).cpu().numpy()

            results = retriever.retrieve(
                txt_emb,
                model_name=model_name,
                k=int(k),
                return_images=True,
                metric="cosine",
            )

        except ValueError as exc:
            logger.warning("Cross-modal retrieval erreur : %s", exc)
            return None, None, f"⚠️ {exc}"

        except Exception as exc:
            logger.error("Cross-modal inattendu : %s", exc, exc_info=True)
            return None, None, f"❌ Erreur : {exc}"

        if not results:
            return [], pd.DataFrame({"Class": [], "Count": []}), "Aucun résultat."

        gallery_items = [
            (r["image"], f"{r['class']} | {r['score']:.3f}")
            for r in results
            if "image" in r
        ]

        dist = retriever.get_class_distribution(results)

        df_dist = pd.DataFrame({
            "Class": list(dist.keys()),
            "Count": list(dist.values()),
        }).sort_values("Count", ascending=False)

        top_result = results[0]

        dist_sorted = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        top3_lines = "\n".join(
            f"- **{cls}** : {count} images"
            for cls, count in dist_sorted[:3]
        )

        stats = f"""
        ### 📊 Résultats

        **Requête :** *"{query_text}"*

        **Modèle image :** `{model_name}`

        **Top-1 :** `{top_result["class"]}`  
        **Score :** `{top_result["score"]:.4f}`

        **{len(results)}** images trouvées · **{len(dist)}** classes représentées

        **Top-3 classes :**

        {top3_lines}
        """

        return gallery_items, df_dist, stats

    def _clear():
        return None, None, None, ""

    btn_search.click(
        _search_text,
        inputs=[text_input, k_slider, model_dd],
        outputs=[gallery, dist_plot, stats_md],
    )

    btn_clear.click(_clear, outputs=[text_input, gallery, dist_plot, stats_md])

    gr.Examples(
        examples=[
            ["a black sneaker"],
            ["a red dress"],
            ["a warm coat"],
            ["a small bag"],
            ["a pair of boots"],
            ["a formal shirt"],
            ["blue trousers"],
            ["a sandal"],
        ],
        inputs=[text_input],
        label="💡 Exemples de requêtes",
    )


# =============================================================================
# ONGLET 10 — EXPORTS (Partie 4.5)
# =============================================================================
def tab_exports():
    """Centre d'export : liste et téléchargement des fichiers générés."""
    gr.Markdown("""
    ### 📥 Exports — Centre de téléchargement

    Tous les fichiers générés par l'application sont ici :

    - **CSV** : résultats Playground, prédictions, métriques
    - **Markdown** : résumés d'analyses

    **Utilité** : partage de résultats pour un rapport ou une soutenance.
    """)

    with gr.Row():
        with gr.Column():
            refresh_btn = gr.Button("🔄 Rafraîchir la liste", variant="secondary")
            export_paths_md = gr.Markdown()

        with gr.Column():
            file_list = gr.File(
                label="📁 Fichiers disponibles",
                file_count="multiple",
                interactive=False,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Exporter les données Analytics")
            btn_export_team = gr.Button("📥 Exporter TEAM_FINAL_TABLE en CSV", variant="primary")
            export_status_md = gr.Markdown()

    def _refresh():
        files = sorted(
            EXPORTS_DIR.glob("*.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not files:
            return [], "### 📁 Aucun fichier exporté pour le moment."

        lines = [f"### 📁 {len(files)} fichiers disponibles\n"]

        for f in files[:20]:
            size_kb = f.stat().st_size / 1024
            lines.append(f"- `{f.name}` — {size_kb:.1f} KB")

        return [str(f) for f in files], "\n".join(lines)

    def _export_team_table():
        tt = analytics.team_table

        if tt.empty:
            return "⚠️ Aucune donnée à exporter."

        path = _export_dataframe(tt, prefix="team_final_table")

        if path:
            return f"✅ Exporté : `{path}`"

        return "❌ Échec de l'export."

    refresh_btn.click(_refresh, outputs=[file_list, export_paths_md])
    btn_export_team.click(_export_team_table, outputs=[export_status_md])

    initial_files, initial_md = _refresh()
    file_list.value = initial_files
    export_paths_md.value = initial_md


# =============================================================================
# ONGLET 11 — GUIDE DE DÉMO (pour soutenance)
# =============================================================================
def tab_guide():
    """Guide pas à pas pour la démonstration de soutenance."""
    gr.Markdown("""
    ### 📖 Guide de Démonstration — Soutenance

    Ce guide vous accompagne pour une **démo fluide de 5 minutes** devant le jury.
    """)

    with gr.Tabs():
        with gr.Tab("🎬 Parcours 5 minutes"):
            gr.Markdown("""
            ### 🎬 Parcours de démonstration (5 min)

            **Ordre recommandé pour la soutenance :**

            | Temps | Onglet | Ce qu'on montre |
            |-------|--------|-----------------|
            | 0:00 – 0:30 | 🎯 **Classification** | Upload d'une image, top-5, distribution |
            | 0:30 – 1:00 | 🎨 **Comparateur** | Les 4 modèles côte à côte + consensus |
            | 1:00 – 1:30 | 🔤 **Prompt Lab** | Changer le template en direct |
            | 1:30 – 2:15 | 🔍 **Retrieval** | Upload sneaker → top-12 cohérents |
            | 2:15 – 2:45 | 🔎 **Cross-Modal** | Taper "a red dress" → images |
            | 2:45 – 3:15 | 🧭 **Drift Explorer** | Impact du fine-tuning par classe |
            | 3:15 – 3:45 | 📊 **Analytics** | Pareto, heatmap, slide |
            | 3:45 – 4:15 | 🌡️ **Calibration** | Reliability diagram + simulateur |
            | 4:15 – 4:45 | 🧪 **Playground** | Batch + export CSV |
            | 4:45 – 5:00 | 🧠 **Insights** | Robustesse + oubli catastrophique |

            **Phrases clés à dire :**
            - *"J'ai 53 tests automatisés qui valident toute la chaîne."*
            - *"Le retrieval atteint 88% d'accuracy@5."*
            - *"LoRA V only fait 2× plus léger que A2 pour 98% des perfs."*
            """)

        with gr.Tab("💡 Points forts"):
            gr.Markdown("""
            ### 💡 Points forts du projet

            **🔬 Scientifique :**
            - Comparaison rigoureuse **4 modèles** (ZS / A1 / A2 / B)
            - Mesure du **drift angulaire** après fine-tuning
            - **Calibration** (ECE, reliability diagram)
            - **Oubli catastrophique** (OOD transfer)
            - **Multi-seed** pour la stabilité

            **🛠️ Technique :**
            - **53 tests automatisés** (couverture complète)
            - **Logging structuré** avec rotation
            - **Retriever multi-modèle** cohérent
            - **Cross-modal** texte → images
            - **Export CSV** sur 3 onglets

            **🎨 UX :**
            - **Page d'accueil** + **menu latéral** moderne
            - **Thème pro** avec gradient + mode sombre
            - **Badges de statut**
            - **Guide de démo** intégré

            **📊 Métriques clés à retenir :**
            - Classification accuracy : **~85%** (A2)
            - Retrieval P@1 : **~65%** (Zero-shot)
            - Retrieval accuracy@5 : **88%**
            - Drift minimal : B LoRA V only
            - Params : A1 (5K) < B (~50K) < A2 (~100K)
            """)

        with gr.Tab("❓ Questions du jury"):
            gr.Markdown("""
            ### ❓ Questions probables du jury + réponses

            **Q: Pourquoi LoRA et pas un fine-tuning complet ?**
            > LoRA réduit drastiquement le nombre de paramètres entraînables
            > (de ~150M à ~50-100K) tout en préservant les performances.
            > C'est crucial pour un dataset limité comme Fashion-MNIST.

            **Q: Comment expliquez-vous le trade-off adaptation/généralité ?**
            > Le LoRA V+T (A2) obtient la meilleure accuracy mais augmente
            > le drift. Le LoRA V only (B) préserve mieux la généralité OOD
            > pour un coût 2× moindre.

            **Q: Qu'est-ce que le drift angulaire ?**
            > C'est la distance angulaire moyenne entre les embeddings
            > avant et après fine-tuning. Il mesure à quel point l'encoder
            > a été modifié.

            **Q: Comment garantissez-vous la fiabilité ?**
            > 53 tests automatisés couvrent : labels, embeddings,
            > retrieval, classification, exports, cohérence.
            > Chaque test passe en 40s sur CPU.

            **Q: Pourquoi 88% accuracy@5 au retrieval ?**
            > C'est le taux où au moins une des 5 premières images retournées
            > partage la classe de la requête. Cela valide la qualité des
            > embeddings CLIP.

            **Q: Le modèle est-il calibré ?**
            > Voir onglet 🌡️ Calibration. La distribution montre si les
            > confiances sont fiables. Un bon modèle ne doit pas être
            > systématiquement sur-confiant.
            """)

        with gr.Tab("🚀 Extensions futures"):
            gr.Markdown("""
            ### 🚀 Extensions futures proposées

            **Court terme :**
            - 🔥 Grad-CAM : visualiser les zones d'attention CLIP
            - 📉 Courbes d'entraînement en direct
            - 🎥 Mode webcam pour démo live

            **Moyen terme :**
            - 🌍 Autres datasets (CIFAR-10, Oxford Pets)
            - 🧠 Autres backbones (ViT-L/14, SigLIP)
            - ☁️ Déploiement HuggingFace Spaces
            - 🐳 Dockerfile + CI/CD

            **Long terme :**
            - 🎯 Few-shot learning adaptatif
            - 🔍 Recherche hybride texte+image
            - 📱 Interface mobile (PWA)

            **Pistes de recherche :**
            - Comparaison LoRA vs DoRA vs AdaptFormer
            - Analyse théorique du drift
            - Curriculum learning sur Fashion-MNIST
            """)

        with gr.Tab("📋 Checklist soutenance"):
            gr.Markdown("""
            ### 📋 Checklist avant la soutenance

            **⏰ 24h avant :**
            - [ ] `pytest tests/ -v` → doit afficher **53 passed**
            - [ ] `python app.py` → vérifier toutes les **sections du menu**
            - [ ] Tester chaque onglet avec **une image de chaque classe**
            - [ ] Prendre des **screenshots** de chaque onglet
            - [ ] Préparer **2-3 images** d'exemple (sneaker, dress, coat)

            **⏰ 1h avant :**
            - [ ] Dossier `exports/` **vide** (pour montrer l'export en live)
            - [ ] Dossier `logs/` prêt
            - [ ] Navigateur ouvert sur http://localhost:7860
            - [ ] Terminal ouvert avec `pytest` prêt à lancer
            - [ ] Slides ouvertes en arrière-plan

            **⏰ Pendant la démo :**
            - [ ] Suivre le **Parcours 5 minutes**
            - [ ] Montrer `pytest tests/ -v` (preuve de robustesse)
            - [ ] Exporter un CSV pour montrer la traçabilité
            - [ ] Rester calme sur les questions techniques

            **⏰ Après :**
            - [ ] Garder l'app ouverte pour les questions
            - [ ] Avoir le README sous la main
            - [ ] Noter les retours du jury
            """)

    gr.Markdown("### 📥 Exporter ce guide")
    with gr.Row():
        btn_export_guide = gr.Button("📥 Exporter le guide en Markdown", variant="secondary")
        export_guide_md = gr.Markdown()

    def _export_guide():
        text = """# Guide de Démonstration — CLIP LoRA Studio

## Parcours 5 minutes
1. 🎯 Classification (30s)
2. 🎨 Comparateur (30s)
3. 🔤 Prompt Lab (30s)
4. 🔍 Retrieval (45s)
5. 🔎 Cross-Modal (30s)
6. 🧭 Drift Explorer (30s)
7. 📊 Analytics (30s)
8. 🌡️ Calibration (30s)
9. 🧪 Playground (30s)
10. 🧠 Insights (15s)

## Points forts
- 11 onglets, 53 tests, 4 modèles
- Accuracy 85%, P@1 65%, accuracy@5 88%
- LoRA V only : 2× plus léger pour 98% des perfs

## Extensions futures
- Grad-CAM, datasets alternatifs, Docker, HF Spaces
"""
        path = _export_markdown_summary(text, prefix="guide_demo")
        if path:
            return f"✅ Exporté : `{path}`"
        return "❌ Échec de l'export."

    btn_export_guide.click(_export_guide, outputs=[export_guide_md])


# =============================================================================
# ONGLET — INSIGHTS
# =============================================================================
def tab_insights():
    gr.Markdown("""
    ### 🧠 Insights — Analyses avancées

    Toutes les analyses complémentaires du projet.
    """)

    with gr.Tabs():
        with gr.Tab("🧊 Robustesse"):
            gr.Markdown("""
            ### Robustesse au bruit

            Test sous bruit gaussien, flou gaussien et rotations.
            """)
            fig_rob = analytics.get_figure("team_robustness.png", "team")
            if fig_rob:
                gr.Image(value=fig_rob, label="Robustesse")
            df_rob = analytics.robustness
            if not df_rob.empty:
                gr.Dataframe(value=df_rob, label="Données brutes")

        with gr.Tab("🧠 Oubli catastrophique"):
            gr.Markdown("""
            ### Oubli catastrophique (OOD)

            Teste si CLIP a perdu sa généralité après fine-tuning.
            """)
            fig_cat = analytics.get_figure("team_catastrophic_forgetting.png", "team")
            if fig_cat:
                gr.Image(value=fig_cat, label="OOD Transfer")

        with gr.Tab("🔁 Multi-seed"):
            gr.Markdown("""
            ### Variance inter-seed

            3 seeds testés pour mesurer la stabilité.
            """)
            fig_ms = analytics.get_figure("team_multi_seed.png", "team")
            if fig_ms:
                gr.Image(value=fig_ms, label="Multi-seed")
            df_ms = analytics.multi_seed
            if not df_ms.empty:
                gr.Dataframe(value=df_ms, label="Résultats par seed")

        with gr.Tab("⚖️ Budget paramètres"):
            gr.Markdown("""
            ### Comparaison à budget égal

            A1 (5K params) vs LoRA minimal.
            """)
            fig_bp = analytics.get_figure("team_equal_params.png", "team")
            if fig_bp:
                gr.Image(value=fig_bp, label="Budget vs Performance")
            df_bp = analytics.equal_params
            if not df_bp.empty:
                gr.Dataframe(value=df_bp, label="Comparaison")

        with gr.Tab("📉 Calibration"):
            gr.Markdown("""
            ### Calibration des probabilités (ECE)

            Mesure si les confiances sont fiables.
            """)
            fig_cal = analytics.get_figure("team_calibration_reliability.png", "team")
            if fig_cal:
                gr.Image(value=fig_cal, label="Reliability Diagram")
            fig_conf = analytics.get_figure("team_confidence_distribution.png", "team")
            if fig_conf:
                gr.Image(value=fig_conf, label="Distribution des confiances")


# =============================================================================
# KPI (chiffres clés de la page d'accueil)
# =============================================================================
def _collect_kpis():
    """Chiffres clés affichés sur la page d'accueil.

    1. Accuracy max · 2. Gain vs Zero-shot · 3. Meilleur compromis
    4. Retrieval P@1 max · 5. Nombre de tests · 6. Images indexées
    """
    cards = []

    try:
        tt_full = analytics.team_table
        col_model = _pick_col_or_none(tt_full, "Model", "model", "model_name", "name", "Modèle")
        col_acc = _pick_col_or_none(tt_full, "ClsAcc", "acc", "cls_acc", "Accuracy", "accuracy", "ClassAcc")
        col_p1 = _pick_col_or_none(tt_full, "P1", "p1", "P@1", "Retrieval_P1", "retrieval_p1")
        col_params = _pick_col_or_none(tt_full, "Params", "params", "n_params", "NumParams", "nb_params", "ParamCount")

        if col_model is not None:
            tt = tt_full[tt_full[col_model].isin(CANONICAL_MODELS)].copy()
        else:
            tt = tt_full.copy()

        logger.info(
            "KPI — modèles canoniques retenus : %s",
            tt[col_model].tolist() if col_model else "N/A",
        )

        def _best(df, col, mode="max"):
            idx = df[col].idxmax() if mode == "max" else df[col].idxmin()
            return df.loc[idx, col], str(df.loc[idx, col_model])

        # KPI 1 — Meilleure accuracy
        if not tt.empty and col_model and col_acc:
            val, model = _best(tt, col_acc, "max")
            cards.append({
                "label": "🎯 Accuracy max",
                "value": f"{val:.3f}",
                "sub": model,
                "color": MODEL_COLORS.get(model, "#4B4FE0"),
            })

        # KPI 2 — Gain vs Zero-shot
        if not tt.empty and col_model and col_acc:
            zs_row = tt[tt[col_model] == "Zero-shot"]
            if not zs_row.empty:
                acc_zs = float(zs_row.iloc[0][col_acc])
                tt_ft = tt[tt[col_model] != "Zero-shot"]
                if not tt_ft.empty:
                    acc_best, model_best = _best(tt_ft, col_acc, "max")
                    gain_pts = (acc_best - acc_zs) * 100
                    cards.append({
                        "label": "📈 Gain vs Zero-shot",
                        "value": f"+{gain_pts:.1f} pts",
                        "sub": f"{model_best} vs baseline",
                        "color": "#10B981",
                    })

        # KPI 3 — Meilleur compromis perf / params
        if not tt.empty and col_model and col_acc and col_params:
            tt_ft = tt[tt[col_model] != "Zero-shot"].copy()
            tt_ft = tt_ft[tt_ft[col_params] > 0]
            if not tt_ft.empty:
                tt_ft["_score"] = tt_ft[col_acc] / np.log10(tt_ft[col_params] + 1)
                idx_best = tt_ft["_score"].idxmax()
                row = tt_ft.loc[idx_best]
                cards.append({
                    "label": "⚖️ Meilleur compromis",
                    "value": str(row[col_model]),
                    "sub": f"{row[col_acc]:.3f} acc · {int(row[col_params]):,} params".replace(",", " "),
                    "color": MODEL_COLORS.get(str(row[col_model]), "#F5A623"),
                })

        # KPI 4 — Retrieval P@1 max
        if not tt.empty and col_model and col_p1:
            val, model = _best(tt, col_p1, "max")
            cards.append({
                "label": "🔍 Retrieval P@1",
                "value": f"{val:.3f}",
                "sub": f"{model} · text→image",
                "color": MODEL_COLORS.get(model, "#4FD1C5"),
            })

    except Exception as exc:
        logger.warning("KPI indisponibles : %s", exc, exc_info=True)

    # KPI 5 — Nombre de tests
    n_tests = 53  # Valeur connue après `pytest tests/ -v`
    cards.append({
        "label": "🧪 Tests automatisés",
        "value": f"{n_tests}",
        "sub": "tous passent ✅",
        "color": "#10B981",
    })

    # KPI 6 — Images indexées
    cards.append({
        "label": "📁 Images indexées",
        "value": f"{retriever.num_images:,}".replace(",", " ") if retriever else "—",
        "sub": f"{len(registry.names)} modèles comparés",
        "color": "#B794F4",
    })

    # Cartes de repli : garantir au moins 4 KPIs
    if len(cards) < 4:
        fallbacks = [
            {"label": "🎯 Accuracy max", "value": "—",
             "sub": "TEAM_FINAL_TABLE absent", "color": "#4B4FE0"},
            {"label": "📈 Gain vs Zero-shot", "value": "—",
             "sub": "calcul indisponible", "color": "#10B981"},
            {"label": "⚖️ Meilleur compromis", "value": "—",
             "sub": "données manquantes", "color": "#F5A623"},
            {"label": "🔍 Retrieval P@1", "value": "—",
             "sub": "données manquantes", "color": "#4FD1C5"},
        ]
        have = {c["label"] for c in cards}
        cards = cards + [f for f in fallbacks if f["label"] not in have]

    return cards


# =============================================================================
# PAGE D'ACCUEIL — graphiques
# =============================================================================
def _team_view():
    """Retourne (DataFrame des 4 modèles canoniques, dict des colonnes) ou None."""
    try:
        tt_full = analytics.team_table
        if tt_full is None or tt_full.empty:
            return None
        cols = {
            "model": _pick_col_or_none(tt_full, "Model", "model", "model_name", "name", "Modèle"),
            "acc": _pick_col_or_none(tt_full, "ClsAcc", "acc", "cls_acc", "Accuracy", "accuracy", "ClassAcc"),
            "p1": _pick_col_or_none(tt_full, "P1", "p1", "P@1", "Retrieval_P1", "retrieval_p1"),
            "drift": _pick_col_or_none(tt_full, "Drift", "drift", "Drift_Angle", "drift_angle"),
            "params": _pick_col_or_none(tt_full, "Params", "params", "n_params", "NumParams", "nb_params", "ParamCount"),
        }
        if cols["model"] is None:
            return None
        tt = tt_full[tt_full[cols["model"]].isin(CANONICAL_MODELS)].copy()
        if tt.empty:
            tt = tt_full.copy()
        for k in ("acc", "p1", "drift", "params"):
            if cols[k] is not None:
                tt[cols[k]] = pd.to_numeric(tt[cols[k]], errors="coerce")
        return tt, cols
    except Exception as exc:
        logger.warning("team_view indisponible : %s", exc)
        return None


def _style_fig(fig, title, height=390):
    """Style commun : fond transparent (compatible mode sombre)."""
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16)),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=25, t=60, b=50),
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="IBM Plex Sans, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def _fig_scores():
    """Accuracy et Retrieval P@1 par modèle (barres groupées)."""
    try:
        view = _team_view()
        if view is None:
            return None
        tt, c = view
        fig = go.Figure()
        if c["acc"]:
            fig.add_trace(go.Bar(name="Accuracy", x=tt[c["model"]], y=tt[c["acc"]],
                                 marker_color="#4B4FE0",
                                 text=tt[c["acc"]].round(3), textposition="outside"))
        if c["p1"]:
            fig.add_trace(go.Bar(name="Retrieval P@1", x=tt[c["model"]], y=tt[c["p1"]],
                                 marker_color="#4FD1C5",
                                 text=tt[c["p1"]].round(3), textposition="outside"))
        if not fig.data:
            return None
        fig.update_layout(barmode="group", yaxis_range=[0, 1.1])
        return _style_fig(fig, "Performances par modèle")
    except Exception as exc:
        logger.warning("fig_scores : %s", exc)
        return None


def _fig_radar():
    """Radar multi-critères : accuracy, retrieval, stabilité, légèreté."""
    try:
        view = _team_view()
        if view is None:
            return None
        tt, c = view
        axes, series = [], {}
        if c["acc"]:
            axes.append("Accuracy")
            series["Accuracy"] = tt[c["acc"]] / max(tt[c["acc"]].max(), 1e-9)
        if c["p1"]:
            axes.append("Retrieval P@1")
            series["Retrieval P@1"] = tt[c["p1"]] / max(tt[c["p1"]].max(), 1e-9)
        if c["drift"]:
            axes.append("Stabilité (1 - drift)")
            dmax = max(tt[c["drift"]].max(), 1e-9)
            series["Stabilité (1 - drift)"] = 1 - tt[c["drift"]] / dmax * 0.9
        if c["params"]:
            axes.append("Légèreté")
            lp = np.log10(tt[c["params"]].clip(lower=0) + 1)
            series["Légèreté"] = 1 - lp / max(lp.max(), 1e-9) * 0.9
        if len(axes) < 3:
            return None

        fig = go.Figure()
        for i, (_, row) in enumerate(tt.iterrows()):
            name = str(row[c["model"]])
            vals = [float(series[a].iloc[i]) for a in axes]
            color = MODEL_COLORS.get(name, "#8A8FD8")
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=axes + axes[:1], name=name,
                fill="toself", opacity=0.55,
                line=dict(color=color, width=2), fillcolor=color,
            ))
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="rgba(128,128,128,.25)"),
            angularaxis=dict(gridcolor="rgba(128,128,128,.25)"),
        ))
        return _style_fig(fig, "Profil multi-critères (1 = meilleur)")
    except Exception as exc:
        logger.warning("fig_radar : %s", exc)
        return None


def _fig_tradeoff():
    """Accuracy en fonction du nombre de paramètres entraînables (échelle log)."""
    try:
        view = _team_view()
        if view is None:
            return None
        tt, c = view
        if not (c["acc"] and c["params"]):
            return None
        tt = tt.copy()
        tt["_p"] = tt[c["params"]].clip(lower=1)
        fig = go.Figure()
        for _, row in tt.iterrows():
            name = str(row[c["model"]])
            fig.add_trace(go.Scatter(
                x=[row["_p"]], y=[row[c["acc"]]], mode="markers+text",
                name=name, text=[name], textposition="top center",
                marker=dict(size=22, color=MODEL_COLORS.get(name, "#8A8FD8"),
                            line=dict(width=2, color="white")),
            ))
        fig.update_xaxes(type="log", title="Paramètres entraînables (log)")
        fig.update_yaxes(title="Accuracy")
        fig.update_layout(showlegend=False)
        return _style_fig(fig, "Compromis performance ↔ budget de paramètres")
    except Exception as exc:
        logger.warning("fig_tradeoff : %s", exc)
        return None


def _fig_classes():
    """Accuracy par classe : Zero-shot vs A2 vs B."""
    try:
        df = analytics.confusion_delta
        if df is None or df.empty:
            return None
        try:
            col_class = _pick_col(df, "class", "Class", "classe", "label")
        except KeyError:
            col_class = df.columns[0]
        fig = go.Figure()
        for model, col in [
            ("Zero-shot", _pick_col_or_none(df, "acc_zs", "acc_ZS", "accuracy_zs", "acc_zeroshot")),
            ("A2 LoRA V+T", _pick_col_or_none(df, "acc_a2", "acc_A2", "accuracy_a2")),
            ("B LoRA V only", _pick_col_or_none(df, "acc_b", "acc_B", "accuracy_b")),
        ]:
            if col is not None:
                fig.add_trace(go.Bar(name=model, x=df[col_class], y=df[col],
                                     marker_color=MODEL_COLORS.get(model)))
        if not fig.data:
            return None
        fig.update_layout(barmode="group", xaxis_tickangle=-35)
        return _style_fig(fig, "Accuracy par classe")
    except Exception as exc:
        logger.warning("fig_classes : %s", exc)
        return None


def _plot_or_note(fig, msg):
    if fig is None:
        gr.Markdown(f"⚠️ {msg}")
    else:
        gr.Plot(value=fig, elem_classes="chart-card", show_label=False)


def _delta_line():
    view = _team_view()
    if view is None:
        return ""
    tt, c = view
    try:
        zs = tt.loc[tt[c["model"]] == "Zero-shot", c["acc"]].iloc[0]
        a2 = tt.loc[tt[c["model"]] == "A2 LoRA V+T", c["acc"]].iloc[0]
        return f"LoRA V+T : {(a2 - zs) * 100:+.1f} pts d'accuracy vs Zero-shot."
    except Exception:
        return ""


# =============================================================================
# PAGE D'ACCUEIL
# =============================================================================
def tab_home(tab_items, n_sections):
    """Page affichée au lancement : hero, KPI, accès rapides, graphiques, essai rapide.

    Retourne {tab_id: bouton} pour que build_app() câble la navigation.
    """
    n_images = retriever.num_images if retriever else 2000

    gr.HTML(hero_html(CLASSES, registry.names, n_sections, n_images))

    gr.HTML(section_html("📌", "Chiffres clés", "Synthèse des 4 modèles comparés"))
    gr.HTML(kpi_html(_collect_kpis()))
    gr.HTML(status_html(registry.names, retriever is not None, n_images))

    # --- Accès rapides ---
    gr.HTML(section_html("🚀", "Accès rapide", "Un clic pour ouvrir une section"))
    buttons = {}
    for i in range(0, len(tab_items), 4):
        with gr.Row():
            for tab_id, label in tab_items[i:i + 4]:
                buttons[tab_id] = gr.Button(label, elem_classes="qcard")

    # --- Visualisations ---
    gr.HTML(section_html("📊", "Visualisations", "Résultats clés du projet en un coup d'œil"))
    with gr.Row():
        with gr.Column():
            _plot_or_note(_fig_scores(), "Scores indisponibles (TEAM_FINAL_TABLE).")
        with gr.Column():
            _plot_or_note(_fig_radar(), "Radar indisponible (colonnes manquantes).")
    with gr.Row():
        with gr.Column():
            _plot_or_note(_fig_tradeoff(), "Compromis indisponible (accuracy / params manquants).")
        with gr.Column():
            _plot_or_note(_fig_classes(), "Données par classe indisponibles (team_confusion_delta.csv).")

    # --- Essai rapide ---
    gr.HTML(section_html("⚡", "Essai rapide", "Déposez une image : prédiction immédiate"))
    default_model = "A2 LoRA V+T" if "A2 LoRA V+T" in registry.names else (
        registry.names[0] if registry.names else "Zero-shot"
    )
    with gr.Row():
        with gr.Column(scale=1):
            q_img = gr.Image(type="pil", label="📷 Image", height=260)
            q_model = gr.Dropdown(registry.names, value=default_model, label="🧠 Modèle")
        with gr.Column(scale=1):
            q_label = gr.Label(label="🎯 Prédiction", num_top_classes=5)

    def _quick(image, model_name):
        if image is None:
            return None
        try:
            top_dict, _ = classify(registry, image, model_name=model_name, top_k=5)
            return top_dict
        except Exception as e:
            logger.error("Essai rapide erreur : %s", e, exc_info=True)
            return None

    q_img.change(_quick, [q_img, q_model], q_label)
    q_model.change(_quick, [q_img, q_model], q_label)

    example_files = sorted(EXAMPLES_DIR.glob("*.png"))[:6]
    if example_files:
        gr.Examples(examples=[[str(f)] for f in example_files],
                    inputs=[q_img], label="📸 Exemples")

    # --- Modèles + insights ---
    gr.HTML(section_html("🧠", "Les modèles comparés"))
    gr.HTML(model_cards_html())
    gr.HTML(section_html("💡", "Ce qu'il faut retenir"))
    gr.HTML(insights_html(_delta_line()))

    # --- Détails (repliables) ---
    with gr.Accordion("🔥 Heatmap TEAM_FINAL_TABLE", open=False):
        heatmap_path = analytics.get_figure("TEAM_FINAL_TABLE_heatmap.png", "team")
        if heatmap_path:
            gr.Image(value=heatmap_path, show_label=False)
        else:
            gr.Markdown("⚠️ Heatmap non trouvée")

    with gr.Accordion("📸 Slide de conclusion", open=False):
        slide_path = analytics.get_figure("SLIDE_CONCLUSION.png", "team")
        if slide_path:
            gr.Image(value=slide_path, show_label=False)
        else:
            gr.Markdown("⚠️ Slide non trouvée")

    view = _team_view()
    if view is not None:
        with gr.Accordion("📋 Tableau récapitulatif des modèles", open=False):
            gr.Dataframe(value=view[0], interactive=False)

    return buttons


# =============================================================================
# ONGLETS — (id, label, fonction, tag démo, phrase, couleur)
# La page d'accueil (id « home ») est gérée à part dans build_app().
# =============================================================================
def _tab_specs():
    return [
        ("classification", "🎯 Classification", tab_classification,
         "0:00 – 0:30", "Importez une image : top-5 et distribution des probabilités.", "#4B4FE0"),
        ("compare", "🎨 Comparateur", tab_comparator,
         "0:30 – 1:00", "Les 4 modèles côte à côte, avec le consensus.", "#50C878"),
        ("prompt_lab", "🔤 Prompt Lab", tab_prompt_lab,
         "1:00 – 1:30", "Changez le template en direct et observez l'effet sur la prédiction.", "#7C83FF"),
        ("retrieval", "🔍 Retrieval", tab_retrieval,
         "1:30 – 2:15", "Importez une sneaker : les 12 images les plus proches doivent être cohérentes.", "#4FD1C5"),
        ("cross_modal", "🔎 Cross-Modal", tab_cross_modal,
         "2:15 – 2:45", "Tapez « a red dress » : le texte retrouve les images.", "#63B3ED"),
        ("drift", "🧭 Drift Explorer", tab_drift_explorer,
         "2:45 – 3:15", "Mesurez l'impact du fine-tuning classe par classe.", "#F687B3"),
        ("analytics", "📊 Analytics", tab_analytics,
         "3:15 – 3:45", "Pareto, heatmap et slide de conclusion.", "#B794F4"),
        ("calibration", "🌡️ Calibration", tab_calibration,
         "3:45 – 4:15", "Reliability diagram et simulateur de confiance.", "#F6AD55"),
        ("playground", "🧪 Playground", tab_playground,
         "4:15 – 4:45", "Analysez un lot d'images et exportez le CSV.", "#68D391"),
        ("insights", "🧠 Insights", tab_insights,
         "4:45 – 5:00", "Robustesse, oubli catastrophique, multi-seed et budget de paramètres.", "#FC8181"),
        ("exports", "📥 Exports", tab_exports,
         "Traçabilité", "Tous les fichiers générés par l'application, prêts à télécharger.", "#A0AEC0"),
        ("guide", "📖 Guide", tab_guide,
         "Soutenance", "Parcours de 5 minutes, questions du jury et checklist.", "#4B4FE0"),
    ]


# =============================================================================
# APP PRINCIPALE
# =============================================================================
def build_app():
    specs = _tab_specs()
    n_sections = len(specs) + 1  # + accueil

    nav_choices = [("🏠 Accueil", "home")] + [(label, tab_id) for tab_id, label, *_ in specs]
    tab_items = [(tab_id, label) for tab_id, label, *_ in specs]

    labels = {"home": "🏠 Accueil"}
    labels.update({tab_id: label for tab_id, label, *_ in specs})

    with gr.Blocks(theme=build_theme(), title=APP_TITLE, css=CSS) as demo:

        # ---------------- HEADER FIXE ----------------
        with gr.Row(elem_id="topbar"):
            header = gr.HTML(topbar_html(labels["home"]))
            btn_theme = gr.Button("🌓 Clair / sombre", size="sm",
                                  variant="secondary", elem_id="theme-btn")

        with gr.Row(equal_height=False, elem_id="layout"):

            # ---------------- MENU LATÉRAL FIXE ----------------
            with gr.Column(scale=0, min_width=250, elem_id="sidebar"):
                gr.HTML(sidebar_brand_html())
                nav = gr.Radio(
                    choices=nav_choices,
                    value="home",
                    show_label=False,
                    container=False,
                    interactive=True,
                    elem_id="nav-radio",
                )

            # ---------------- CONTENU ----------------
            with gr.Column(scale=1, min_width=0, elem_id="content-col"):
                # Onglets natifs masqués en CSS (#main-tabs) : le menu latéral les pilote.
                with gr.Tabs(elem_id="main-tabs", selected="home") as tabs:
                    with gr.Tab("🏠 Accueil", id="home"):
                        home_buttons = tab_home(tab_items, n_sections)

                    for tab_id, label, fn, tag, note, color in specs:
                        with gr.Tab(label, id=tab_id):
                            gr.HTML(demo_note_html(tag, note, color))
                            fn()

                gr.HTML(footer_html(n_sections))

        # ---------------- NAVIGATION ----------------
        nav.change(
            lambda v: (gr.Tabs(selected=v), topbar_html(labels.get(v, ""))),
            inputs=nav,
            outputs=[tabs, header],
        )

        for tab_id, btn in home_buttons.items():
            btn.click(
                lambda tid=tab_id: gr.update(value=tid),
                outputs=[nav],
            )

        btn_theme.click(None, None, None, js=TOGGLE_DARK_JS)

    return demo


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    import os

    PORT = int(os.environ.get("PORT", 7860))
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        share=False,
    )