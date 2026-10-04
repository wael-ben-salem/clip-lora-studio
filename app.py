"""
🎨 CLIP LoRA Studio — Application Gradio
Étude du fine-tuning de CLIP avec LoRA sur Fashion-MNIST.

Phase 4 — Les 5 onglets fonctionnels
"""

import gradio as gr
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Modules internes ---
from modules.models import ModelRegistry
from modules.inference import classify, classify_multi
from modules.analytics import Analytics
from utils.constants import (
    CLASSES, CLASS_PROMPTS, EXAMPLES_DIR, FIGURES_DIR, DATA_DIR
)


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

    # 1) exact
    for c in candidates:
        if c in df.columns:
            return c

    # 2) case-insensitive
    lower = {str(col).lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    # 3) loose (ignore _ / spaces / -)
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
# INITIALISATION (au démarrage de l'app)
# =============================================================================
print("=" * 70)
print("🚀 INITIALISATION CLIP LoRA Studio")
print("=" * 70)

registry  = ModelRegistry().load_all()
analytics = Analytics()

print(f"\n✅ Prêt ! {len(registry.names)} modèles disponibles : {registry.names}")
print(f"✅ Analytics : {len(analytics.list_available_figures('team'))} figures, "
      f"{len(analytics.list_available_data())} fichiers de données")

# Debug : afficher les colonnes des tables clés au démarrage
try:
    if not analytics.team_table.empty:
        print(f"🔍 team_table columns : {list(analytics.team_table.columns)}")
except Exception as e:
    print(f"⚠️  team_table indisponible : {e}")

try:
    if not analytics.confusion_delta.empty:
        print(f"🔍 confusion_delta columns : {list(analytics.confusion_delta.columns)}")
except Exception as e:
    print(f"⚠️  confusion_delta indisponible : {e}")


# =============================================================================
# CONFIGURATION UI
# =============================================================================
APP_TITLE = "🎨 CLIP LoRA Studio"
APP_DESCRIPTION = """
**Explorez, comparez, comprenez** comment LoRA fine-tune CLIP sur Fashion-MNIST.

Projet académique — étude du trade-off **adaptation ↔ préservation de la généralité**.
"""

COLORS = {
    "Zero-shot":      "#4A90E2",
    "A2 LoRA V+T":    "#50C878",
    "B LoRA V only":  "#F5A623",
}


# =============================================================================
# ONGLET 1 — CLASSIFICATION (Phase 3)
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
        if image is None:
            return None, None, pd.DataFrame()
        top_dict, df = classify(registry, image, model_name=model_name, top_k=k)
        df_plot = df.head(k).copy()
        return top_dict, df_plot, df
    
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
# ONGLET 2 — PROMPT LAB (Phase 3)
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
        
        top_dict, df = classify(registry, image, model_name=model_name,
                                 top_k=5, custom_prompts=prompts)
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
# ONGLET 3 — DRIFT EXPLORER (Phase 4) 🆕
# =============================================================================
def tab_drift_explorer():
    gr.Markdown("""
    ### 🧭 Drift Explorer — Exploration interactive du drift
    
    Le **drift angulaire** mesure à quel point les embeddings ont été modifiés 
    par le fine-tuning.
    
    **Interprétation** :
    - `drift ≈ 0` → encoder inchangé
    - `drift > 0.5` → encoder fortement modifié
    """)
    
    # Charger les données
    df_delta = analytics.confusion_delta
    
    if df_delta.empty:
        gr.Markdown("⚠️ Données non trouvées. Vérifie `assets/data/team_confusion_delta.csv`.")
        return
    
    # Résoudre les colonnes (robuste aux variations de nommage)
    try:
        col_class = _pick_col(df_delta, "class", "Class", "classe", "label")
    except KeyError:
        col_class = df_delta.columns[0]  # fallback : première colonne

    col_delta     = _pick_col_or_none(df_delta, "delta_a2_b", "delta_A2_B", "deltaA2B")
    col_acc_a2    = _pick_col_or_none(df_delta, "acc_a2", "acc_A2", "accuracy_a2")
    col_acc_zs    = _pick_col_or_none(df_delta, "acc_zs", "acc_ZS", "accuracy_zs", "acc_zeroshot")
    col_acc_b     = _pick_col_or_none(df_delta, "acc_b",  "acc_B",  "accuracy_b")
    col_delta_zs  = _pick_col_or_none(df_delta, "delta_a2_zs", "delta_A2_ZS", "deltaA2ZS")
    
    # KPIs
    with gr.Row():
        if col_delta is not None:
            best_class_row = df_delta.loc[df_delta[col_delta].idxmax()]
            worst_class_row = df_delta.loc[df_delta[col_delta].idxmin()]
            gr.Markdown(f"""
            ### 📊 Points clés
            - 🏆 **Classe la plus améliorée par A2 vs B** : `{best_class_row[col_class]}` 
              ({best_class_row[col_delta]*100:+.2f} pts)
            - ⚠️ **Classe la moins améliorée** : `{worst_class_row[col_class]}` 
              ({worst_class_row[col_delta]*100:+.2f} pts)
            """)
        else:
            gr.Markdown("### 📊 Points clés\n⚠️ Colonne `delta_a2_b` introuvable.")
    
    # Contrôles — construire la liste des métriques réellement disponibles
    metric_choices = []
    for name, col in [
        ("acc_zs", col_acc_zs), ("acc_a2", col_acc_a2), ("acc_b", col_acc_b),
        ("delta_a2_b", col_delta), ("delta_a2_zs", col_delta_zs),
    ]:
        if col is not None:
            metric_choices.append(name)
    
    if not metric_choices:
        gr.Markdown("⚠️ Aucune métrique reconnue dans `confusion_delta`.")
        gr.Dataframe(value=df_delta, label="Données brutes", interactive=False)
        return
    
    default_metric = "acc_a2" if "acc_a2" in metric_choices else metric_choices[0]
    
    with gr.Row():
        metric_dd = gr.Dropdown(
            choices=metric_choices,
            value=default_metric,
            label="📊 Métrique à afficher",
        )
        sort_dd = gr.Dropdown(
            choices=["Ordre classes", "Décroissant", "Croissant"],
            value="Ordre classes",
            label="🔃 Tri",
        )
    
    # Graphique interactif
    plot = gr.Plot(label="📈 Graphique interactif")
    
    # Mapping nom logique -> nom de colonne réel
    metric_to_col = {
        "acc_zs": col_acc_zs, "acc_a2": col_acc_a2, "acc_b": col_acc_b,
        "delta_a2_b": col_delta, "delta_a2_zs": col_delta_zs,
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
            df_plot, x=col_class, y=col,
            color=col,
            color_continuous_scale="RdYlGn",
            title=f"{metric} par classe",
            labels={col_class: "Classe", col: metric},
            height=500,
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="white",
            font=dict(size=12),
        )
        fig.update_traces(marker_line_color="black", marker_line_width=1)
        return fig
    
    plot.value = make_plot(default_metric, "Ordre classes")
    
    metric_dd.change(make_plot, [metric_dd, sort_dd], plot)
    sort_dd.change(make_plot, [metric_dd, sort_dd], plot)
    
    # Tableau détaillé
    gr.Markdown("### 📋 Tableau détaillé")
    gr.Dataframe(value=df_delta, label="Données brutes", interactive=False)
    
    # Graphique comparatif
    gr.Markdown("### 📊 Comparaison ZS vs A2 vs B")
    
    def make_comparison():
        fig = go.Figure()
        for model, col, color in [
            ("Zero-shot",    col_acc_zs, COLORS["Zero-shot"]),
            ("A2 LoRA V+T",  col_acc_a2, COLORS["A2 LoRA V+T"]),
            ("B LoRA V only",col_acc_b,  COLORS["B LoRA V only"]),
        ]:
            if col is not None:
                fig.add_trace(go.Bar(
                    name=model, x=df_delta[col_class], y=df_delta[col],
                    marker_color=color, marker_line_color="black",
                    marker_line_width=1,
                ))
        fig.update_layout(
            barmode="group", height=500, xaxis_tickangle=-45,
            plot_bgcolor="white", title="Accuracy par classe",
            yaxis_title="Accuracy",
        )
        return fig
    
    gr.Plot(value=make_comparison())


# =============================================================================
# ONGLET 4 — ANALYTICS (Phase 4) 🆕
# =============================================================================
def tab_analytics():
    gr.Markdown("""
    ### 📈 Analytics — Dashboard récapitulatif du projet
    
    Vue d'ensemble des 4 modèles comparés avec leurs performances clés.
    """)
    
    # Charger les données
    tt = analytics.team_table
    
    if tt.empty:
        gr.Markdown("⚠️ TEAM_FINAL_TABLE non trouvé.")
        return
    
    # --- Résolution robuste des colonnes ---
    # team_table réelle : ['model', 'acc', 'f1', 'p1', 'p5', 'map_', 't1', 'mrr', 'drift', 'params']
    try:
        col_model  = _pick_col(tt, "Model",  "model", "model_name", "name", "Modèle")
        col_acc    = _pick_col(tt, "ClsAcc", "acc", "cls_acc", "Cls_Acc",
                                "Accuracy", "accuracy", "clsacc", "ClassAcc")
        col_p1     = _pick_col(tt, "P1",     "p1", "P@1", "Retrieval_P1",
                                "recall@1", "Recall1", "retrieval_p1")
        col_drift  = _pick_col(tt, "Drift",  "drift", "Drift_Angle", "drift_angle")
        col_params = _pick_col(tt, "Params", "params", "n_params", "NumParams",
                                "nb_params", "ParamCount")
    except KeyError as e:
        gr.Markdown(f"⚠️ Colonnes manquantes dans TEAM_FINAL_TABLE : `{e}`")
        gr.Dataframe(value=tt, label="Données brutes (colonnes détectées)",
                     interactive=False)
        return
    
    # KPIs
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            ### 🎯 Accuracy
            **{tt[col_acc].max():.3f}** ({tt.loc[tt[col_acc].idxmax(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 🔍 Retrieval P@1
            **{tt[col_p1].max():.3f}** ({tt.loc[tt[col_p1].idxmax(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 🧭 Drift minimal
            **{tt[col_drift].min():.3f}** ({tt.loc[tt[col_drift].idxmin(), col_model]})
            """)
        with gr.Column():
            gr.Markdown(f"""
            ### 💾 Params max
            **{tt[col_params].max():,}**
            """)
    
    # Tabs
    with gr.Tabs():
        with gr.Tab("📊 Tableau"):
            gr.Dataframe(value=tt, label="TEAM_FINAL_TABLE", interactive=False)
            
            # Tri par accuracy
            ranked = tt.sort_values(col_acc, ascending=False)
            gr.Dataframe(value=ranked, label="Classement par accuracy",
                          interactive=False)
        
        with gr.Tab("🔥 Heatmap"):
            heatmap_path = analytics.get_figure("TEAM_FINAL_TABLE_heatmap.png", "team")
            if heatmap_path:
                gr.Image(value=heatmap_path, label="Heatmap TEAM_FINAL_TABLE")
            else:
                gr.Markdown("⚠️ Heatmap non trouvée")
        
        with gr.Tab("🎯 Pareto"):
            # Scatter interactif
            fig = px.scatter(
                tt, x=col_acc, y=col_p1,
                size=col_params, color=col_drift,
                hover_name=col_model, text=col_model,
                color_continuous_scale="RdYlGn_r",
                title="Trade-off Accuracy ↔ Retrieval (taille = params)",
                height=500,
            )
            fig.update_traces(
                marker=dict(line=dict(width=2, color='black')),
                textposition="top center",
            )
            fig.update_layout(
                xaxis_title="Classification Accuracy →",
                yaxis_title="Text→Image P@1 →",
                plot_bgcolor="white",
            )
            gr.Plot(value=fig)
    
    # Slide de conclusion
    gr.Markdown("### 📸 Slide de conclusion")
    slide_path = analytics.get_figure("SLIDE_CONCLUSION.png", "team")
    if slide_path:
        gr.Image(value=slide_path, label="Synthèse du projet")
    else:
        gr.Markdown("⚠️ Slide non trouvée")
    
    # Message d'insight — calcul robuste du delta ZS → A2
    try:
        acc_zs = tt.loc[tt[col_model] == 'Zero-shot', col_acc].values
        acc_a2 = tt.loc[tt[col_model] == 'A2 LoRA V+T', col_acc].values
        if len(acc_zs) and len(acc_a2):
            delta_pts = (acc_a2[0] - acc_zs[0]) * 100
            insight_line = (f"- ✅ **LoRA V+T** : {delta_pts:+.1f}% accuracy "
                            f"vs Zero-shot")
        else:
            insight_line = "- ✅ **LoRA V+T** : comparaison ZS/A2 indisponible"
    except Exception:
        insight_line = "- ✅ **LoRA V+T** : comparaison ZS/A2 indisponible"
    
    gr.Markdown(f"""
    ---
    ### 💡 Insights clés
    
    {insight_line}
    - ✅ **LoRA V only** : 2× plus léger avec 98% des performances d'A2
    - ⚡ **A1 LinearHead** : efficace à budget ultra-réduit (5K params)
    """)


# =============================================================================
# ONGLET 5 — INSIGHTS (Phase 4) 🆕
# =============================================================================
def tab_insights():
    gr.Markdown("""
    ### 🧠 Insights — Analyses avancées
    
    Toutes les analyses complémentaires du projet.
    """)
    
    with gr.Tabs():
        # --- Tab 1 : Robustesse ---
        with gr.Tab("🧊 Robustesse"):
            gr.Markdown("""
            #### Robustesse au bruit
            Test sous bruit gaussien, flou gaussien et rotations.
            """)
            fig_rob = analytics.get_figure("team_robustness.png", "team")
            if fig_rob:
                gr.Image(value=fig_rob, label="Robustesse")
            df_rob = analytics.robustness
            if not df_rob.empty:
                gr.Dataframe(value=df_rob, label="Données brutes")
        
        # --- Tab 2 : Oubli catastrophique ---
        with gr.Tab("🧠 Oubli catastrophique"):
            gr.Markdown("""
            #### Oubli catastrophique (OOD)
            Teste si CLIP a perdu sa généralité après fine-tuning.
            """)
            fig_cat = analytics.get_figure("team_catastrophic_forgetting.png", "team")
            if fig_cat:
                gr.Image(value=fig_cat, label="OOD Transfer")
        
        # --- Tab 3 : Multi-seed ---
        with gr.Tab("🔁 Multi-seed"):
            gr.Markdown("""
            #### Variance inter-seed
            3 seeds testés pour mesurer la stabilité.
            """)
            fig_ms = analytics.get_figure("team_multi_seed.png", "team")
            if fig_ms:
                gr.Image(value=fig_ms, label="Multi-seed")
            df_ms = analytics.multi_seed
            if not df_ms.empty:
                gr.Dataframe(value=df_ms, label="Résultats par seed")
        
        # --- Tab 4 : Budget paramètres ---
        with gr.Tab("⚖️ Budget paramètres"):
            gr.Markdown("""
            #### Comparaison à budget égal
            A1 (5K params) vs LoRA minimal.
            """)
            fig_bp = analytics.get_figure("team_equal_params.png", "team")
            if fig_bp:
                gr.Image(value=fig_bp, label="Budget vs Performance")
            df_bp = analytics.equal_params
            if not df_bp.empty:
                gr.Dataframe(value=df_bp, label="Comparaison")
        
        # --- Tab 5 : Calibration ---
        with gr.Tab("📉 Calibration"):
            gr.Markdown("""
            #### Calibration des probabilités (ECE)
            Mesure si les confiances sont fiables.
            """)
            fig_cal = analytics.get_figure("team_calibration_reliability.png", "team")
            if fig_cal:
                gr.Image(value=fig_cal, label="Reliability Diagram")
            fig_conf = analytics.get_figure("team_confidence_distribution.png", "team")
            if fig_conf:
                gr.Image(value=fig_conf, label="Distribution des confiances")


# =============================================================================
# APP PRINCIPALE
# =============================================================================
def build_app():
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        title=APP_TITLE,
        css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
        h1 { color: #4A4A8A; }
        h2 { color: #5B5BB0; }
        h3 { color: #6C6CC5; }
        """,
    ) as demo:
        
        # --- En-tête ---
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        
        # --- 5 Onglets ---
        with gr.Tabs():
            with gr.Tab("🎯 Classification"):  tab_classification()
            with gr.Tab("🔤 Prompt Lab"):      tab_prompt_lab()
            with gr.Tab("🧭 Drift Explorer"):  tab_drift_explorer()
            with gr.Tab("📊 Analytics"):       tab_analytics()
            with gr.Tab("🧠 Insights"):        tab_insights()
        
        # --- Footer ---
        gr.Markdown("""
        ---
        ### 📚 À propos
        
        Projet académique étudiant le fine-tuning de CLIP avec LoRA sur Fashion-MNIST.
        
        **Modèles** : Zero-shot · A1 LinearHead · A2 LoRA V+T · B LoRA V only
        
        **Statut** : ✅ Phase 4 — Tous les onglets fonctionnels
        """)
    
    return demo


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    demo = build_app()
    PORT = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        share=False,
    )