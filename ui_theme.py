"""
ui_theme.py — Couche de présentation de CLIP LoRA Studio.

Ce module ne contient AUCUNE logique métier : uniquement le thème Gradio,
le CSS et les blocs HTML (hero, KPI, sidebar, sections, footer).

Nouveautés :
  - menu latéral moderne (#sidebar + #nav-radio) qui pilote les onglets
  - page d'accueil (hero, KPI, accès rapides, graphiques, essai rapide)
  - onglets natifs masqués (#main-tabs) : la navigation passe par le menu
"""

from __future__ import annotations

import html
import math
import random
from typing import Dict, Iterable, List

import gradio as gr

# =============================================================================
# PALETTE (cohérente avec COLORS de app.py : ZS bleu · A2 vert · B orange)
# =============================================================================
MODEL_COLORS: Dict[str, str] = {
    "Zero-shot": "#4A90E2",
    "A1 LinearHead": "#8A8FD8",
    "A2 LoRA V+T": "#50C878",
    "B LoRA V only": "#F5A623",
}

# Une teinte par classe Fashion-MNIST (10 classes)
CLASS_PALETTE = [
    "#7C83FF", "#4FD1C5", "#F6AD55", "#F687B3", "#68D391",
    "#63B3ED", "#FC8181", "#B794F4", "#F6E05E", "#A0AEC0",
]

FONTS_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700"
    "&family=IBM+Plex+Mono:wght@500;600&display=swap');"
)

# JS du bouton clair / sombre
TOGGLE_DARK_JS = "() => { document.body.classList.toggle('dark'); }"


# =============================================================================
# THÈME GRADIO
# =============================================================================
def build_theme() -> gr.themes.Base:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )
    try:
        theme = theme.set(
            body_background_fill="#F4F5FA",
            body_background_fill_dark="#0E1020",
            block_background_fill="#FFFFFF",
            block_background_fill_dark="#171A33",
            block_border_width="1px",
            block_border_color="#E3E5F0",
            block_border_color_dark="#2A2E52",
            block_shadow="0 1px 2px rgba(18,20,43,.04)",
            block_title_text_weight="600",
            block_label_text_weight="600",
            button_primary_background_fill="#4B4FE0",
            button_primary_background_fill_hover="#3A3DB8",
            button_primary_text_color="#FFFFFF",
        )
    except Exception:  # un token inconnu ne doit jamais empêcher l'app de démarrer
        pass
    return theme


# =============================================================================
# CSS
# =============================================================================
CSS = FONTS_IMPORT + """

/* ---------- Tokens ---------- */
:root {
    --ink: #12142B;
    --paper: #F4F5FA;
    --panel: #FFFFFF;
    --line: #E3E5F0;
    --muted: #5F647E;
    --indigo: #4B4FE0;
    --indigo-deep: #3A3DB8;
    --zs: #4A90E2;
    --a1: #8A8FD8;
    --a2: #50C878;
    --b: #F5A623;
    --display: 'Bricolage Grotesque', 'IBM Plex Sans', ui-sans-serif, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
}
.dark {
    --ink: #EDEEFF;
    --paper: #0E1020;
    --panel: #171A33;
    --line: #2A2E52;
    --muted: #A3A8CC;
}

/* ---------- Structure générale ---------- */
.gradio-container {
    max-width: 1560px !important;
    margin: 0 auto !important;
    background: var(--paper) !important;
}
footer { display: none !important; }

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
    font-family: var(--display) !important;
    letter-spacing: -0.01em;
}
.gradio-container h3 {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    margin-top: .2em;
}
.gradio-container h4 { color: var(--ink) !important; font-weight: 600 !important; }
.gradio-container h2 { color: var(--ink) !important; font-weight: 700 !important; }
.gradio-container .prose p,
.gradio-container .prose li { line-height: 1.6; max-width: 78ch; }
.gradio-container .prose strong { color: var(--ink); }

/* =====================================================================
   MENU LATÉRAL  (Column #sidebar + Radio #nav-radio)
   ===================================================================== */
/* ---------- HEADER FIXE ---------- */
.gradio-container { padding-top: 78px !important; }

#topbar {
    position: fixed !important; top: 0; left: 0; right: 0; z-index: 1000;
    height: 62px; margin: 0 !important; padding: 0 22px !important;
    display: flex !important; flex-wrap: nowrap !important;
    align-items: center !important; gap: 12px !important;
    background: color-mix(in srgb, var(--panel) 90%, transparent) !important;
    -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
    border: none !important; border-bottom: 1px solid var(--line) !important;
    box-shadow: 0 4px 18px rgba(18,20,43,.06);
}
#topbar > * { min-width: 0 !important; }
#topbar > :first-child { flex: 1 1 auto !important; }
#topbar > :not(:first-child) { flex: 0 0 auto !important; width: auto !important; }
#topbar .block, #topbar .prose, #topbar .html-container {
    border: none !important; background: transparent !important;
    padding: 0 !important; box-shadow: none !important; margin: 0 !important;
}
.topbar-in { display: flex; align-items: center; gap: 14px; flex-wrap: nowrap; }
.topbar-in .logo {
    width: 38px; height: 38px; border-radius: 12px; flex: none;
    display: grid; place-items: center; font-size: 1.2rem; color: #fff;
    background: linear-gradient(135deg, #4B4FE0 0%, #8B5CF6 100%);
    box-shadow: 0 6px 16px rgba(75,79,224,.35);
}
.topbar-in .t {
    font-family: var(--display); font-weight: 700; font-size: 1.05rem;
    color: var(--ink); line-height: 1.1; white-space: nowrap;
}
.topbar-in .s { font-size: .74rem; color: var(--muted); white-space: nowrap; }
.topbar-in .crumb {
    margin-left: 10px; padding: 6px 14px; border-radius: 999px;
    font-weight: 600; font-size: .9rem; white-space: nowrap;
    color: var(--indigo);
    background: color-mix(in srgb, var(--indigo) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--indigo) 30%, transparent);
}
#theme-btn { min-width: 0 !important; white-space: nowrap; }

/* ---------- MENU LATÉRAL FIXE (défile seul, toujours visible) ---------- */
#layout { align-items: flex-start !important; }
#sidebar {
    position: fixed !important;
    top: 78px; bottom: 14px;
    left: max(14px, calc((100vw - 1560px) / 2 + 14px));
    width: 250px !important; min-width: 0 !important; max-width: 250px !important;
    flex: none !important;
    display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    z-index: 50;
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 16px 12px !important;
    gap: 8px !important;
    box-shadow: 0 8px 28px rgba(18,20,43,.06);
}
#sidebar > * { width: 100% !important; min-width: 0 !important; max-width: 100% !important; flex: none !important; }
#sidebar::-webkit-scrollbar { width: 6px; }
#sidebar::-webkit-scrollbar-thumb { background: var(--line); border-radius: 6px; }
#content-col { margin-left: 280px !important; min-width: 0 !important; }
.side-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 8px 14px 8px; margin-bottom: 4px;
    border-bottom: 1px solid var(--line);
}
.side-brand .logo {
    width: 42px; height: 42px; border-radius: 13px; flex: none;
    display: grid; place-items: center; font-size: 1.3rem; color: #fff;
    background: linear-gradient(135deg, #4B4FE0 0%, #8B5CF6 100%);
    box-shadow: 0 6px 16px rgba(75,79,224,.35);
}
.side-brand .t { font-family: var(--display); font-weight: 700; font-size: 1.05rem; color: var(--ink); line-height: 1.15; }
.side-brand .s { font-size: .75rem; color: var(--muted); margin-top: 2px; }
.side-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
    color: var(--muted); padding: 8px 12px 2px 12px;
}

#nav-radio, #nav-radio fieldset, #nav-radio .wrap {
    width: 100% !important; min-width: 0 !important; max-width: 100% !important;
}
#nav-radio {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}
#nav-radio .wrap {
    display: flex !important; flex-direction: column !important;
    gap: 3px !important; flex-wrap: nowrap !important;
}
#nav-radio label {
    display: flex !important; align-items: center; width: 100%; box-sizing: border-box;
    padding: 10px 14px !important; border-radius: 12px !important;
    border: 1px solid transparent !important;
    background: transparent !important; box-shadow: none !important;
    color: var(--muted); font-weight: 600; font-size: .95rem; cursor: pointer;
    transition: background .15s ease, color .15s ease, transform .15s ease;
}
#nav-radio label span { color: inherit !important; }
#nav-radio label:hover {
    background: color-mix(in srgb, var(--indigo) 9%, transparent) !important;
    color: var(--ink); transform: translateX(2px);
}
#nav-radio label.selected,
#nav-radio label:has(input:checked) {
    background: linear-gradient(135deg, #4B4FE0 0%, #6D5BF0 100%) !important;
    color: #fff !important;
    box-shadow: 0 6px 16px rgba(75,79,224,.30) !important;
}
#nav-radio input[type="radio"] {
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}

/* Onglets natifs masqués : la navigation passe par le menu latéral.
   (sélecteurs « enfant direct » : les onglets imbriqués restent visibles) */
#main-tabs > .tab-nav,
#main-tabs > .tab-wrapper { display: none !important; }
#main-tabs > .tabitem,
#main-tabs > div[role="tabpanel"] { border: none !important; padding: 0 !important; }

/* ---------- Hero ---------- */
.hero {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
    gap: 28px;
    align-items: center;
    padding: 34px 38px;
    border-radius: 22px;
    color: #F3F4FF;
    background-color: #10122A;
    background-image:
        radial-gradient(900px 300px at 8% -10%, rgba(99,102,241,.35), transparent 60%),
        radial-gradient(rgba(255,255,255,.07) 1px, transparent 1px);
    background-size: auto, 22px 22px;
    overflow: hidden;
}
.hero h1 {
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: clamp(1.8rem, 3vw, 2.7rem) !important;
    line-height: 1.08 !important;
    color: #FFFFFF !important;
    margin: 0 0 14px 0 !important;
    letter-spacing: -0.025em;
    border: none !important;
}
.hero .brand {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: var(--display); font-weight: 600; font-size: 1rem;
    color: #C9CCFF; margin-bottom: 18px;
}
.hero .brand b {
    width: 30px; height: 30px; border-radius: 9px;
    display: grid; place-items: center;
    background: #4B4FE0; color: #fff; font-size: 1rem;
}
.hero p.lead {
    color: rgba(236,238,255,.86);
    font-size: 1.05rem; line-height: 1.6; margin: 0 0 20px 0; max-width: 52ch;
}
.hero .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
.status-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: .85em; font-weight: 600; margin-right: 0;
}
.badge-success { background: #10B981; color: #fff; }
.badge-info    { background: #3B82F6; color: #fff; }
.badge-warning { background: #F59E0B; color: #fff; }

.hero .legend { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 12px; border-radius: 999px; font-size: .82rem;
    border: 1px solid rgba(255,255,255,.2); color: #E8EAFF;
}
.chip i {
    width: 9px; height: 9px; border-radius: 50%; background: var(--c, #fff);
    display: inline-block;
}

/* Constellation d'embeddings */
.hero figure { margin: 0; }
.hero svg.cstl { width: 100%; height: auto; display: block; }
.hero figcaption {
    font-size: .8rem; color: rgba(220,224,255,.7); margin-top: 6px; text-align: right;
}
.cstl circle {
    opacity: 0;
    animation: gather 1.9s cubic-bezier(.22,.8,.25,1) forwards;
}
.cstl text {
    opacity: 0; fill: #E8EAFF; font-size: 11px;
    font-family: var(--display); font-weight: 600;
    animation: reveal .6s ease 2.1s forwards;
}
.cstl ellipse { fill: none; stroke: rgba(255,255,255,.1); stroke-dasharray: 3 7; }
@keyframes gather {
    from { transform: translate(var(--sx), var(--sy)); opacity: 0; }
    15%  { opacity: .9; }
    to   { transform: translate(0, 0); opacity: 1; }
}
@keyframes reveal { to { opacity: .92; } }

/* ---------- Titres de section (accueil) ---------- */
.sec { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; margin: 26px 0 10px 0; }
.sec .ic {
    width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center;
    background: color-mix(in srgb, var(--indigo) 12%, transparent); font-size: 1rem;
    align-self: center;
}
.sec h3 { margin: 0 !important; }
.sec .sub { color: var(--muted); font-size: .9rem; }

/* ---------- KPI ---------- */
.kpis {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin: 16px 0 6px 0;
}
.kpi {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 20px;
    border-left: 5px solid var(--c, var(--indigo));
    transition: transform .15s ease, box-shadow .15s ease;
}
.kpi:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(18,20,43,.08);
}
.kpi .v {
    font-family: var(--mono); font-weight: 600;
    font-size: 1.7rem; line-height: 1.1; color: var(--c, var(--ink));
}
.kpi .l {
    font-size: .9rem; color: var(--muted); margin-top: 6px;
    font-weight: 500; letter-spacing: -0.005em;
}
.kpi .s {
    font-size: .85rem; color: var(--ink); margin-top: 2px; font-weight: 600;
}

/* Anciennes cartes (compatibilité) */
.metric-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 16px; text-align: center;
}
.metric-value { font-size: 1.8em; font-weight: 700; color: var(--ink); font-family: var(--mono); }
.metric-label { font-size: .85em; color: var(--muted); }

/* ---------- Barre d'état ---------- */
.statusbar {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px;
    padding: 10px 16px; margin: 10px 0 4px 0;
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    font-size: .9rem; color: var(--muted);
}
.statusbar .grp { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.statusbar .chip { color: var(--ink); border-color: var(--line); }
.statusbar .ok  { color: #0E9F6E; font-weight: 600; }
.statusbar .bad { color: #D97706; font-weight: 600; }

/* ---------- Accès rapides (boutons de l'accueil) ---------- */
.qcard {
    min-height: 62px !important;
    border-radius: 14px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-weight: 600 !important; font-size: .98rem !important;
    border: 1px solid var(--line) !important;
    background: var(--panel) !important; color: var(--ink) !important;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.qcard:hover {
    transform: translateY(-2px);
    border-color: var(--indigo) !important;
    box-shadow: 0 8px 20px rgba(75,79,224,.16);
}

/* ---------- Cartes graphiques ---------- */
.chart-card {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 6px !important;
}
.gradio-container .js-plotly-plot text { fill: var(--ink) !important; }

/* ---------- Cartes modèles ---------- */
.models {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px; margin: 10px 0;
}
.mcard {
    border: 1px solid var(--line); border-top: 4px solid var(--c);
    border-radius: 12px; padding: 12px 14px; background: var(--panel);
    transition: transform .15s ease, box-shadow .15s ease;
}
.mcard:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(18,20,43,.08); }
.mcard b { display: block; color: var(--ink); font-family: var(--display); }
.mcard span { font-size: .85rem; color: var(--muted); }

/* ---------- Insight cards ---------- */
.insights {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px; margin: 10px 0;
}
.insight {
    display: flex; gap: 12px; padding: 14px 16px; border-radius: 14px;
    background: var(--panel); border: 1px solid var(--line);
}
.insight .ic { font-size: 1.5rem; line-height: 1; }
.insight b { color: var(--ink); display: block; font-family: var(--display); }
.insight span { color: var(--muted); font-size: .88rem; line-height: 1.5; }

/* Bandeau « à montrer en démo » */
.demo-note {
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    padding: 12px 16px; margin: 4px 0 14px 0;
    border-radius: 12px; border: 1px solid var(--line);
    border-left: 5px solid var(--c, var(--indigo));
    background: var(--panel);
    color: var(--ink); font-size: .95rem;
}
.demo-note .tag {
    font-family: var(--mono); font-weight: 600; font-size: .8rem;
    padding: 3px 10px; border-radius: 8px; white-space: nowrap;
    background: color-mix(in srgb, var(--c, #4B4FE0) 14%, transparent);
    color: var(--ink);
}

/* ---------- Onglets imbriqués (Guide, Insights, Analytics) ---------- */
.gradio-container button[role="tab"] {
    white-space: nowrap; font-weight: 600 !important; font-size: .95em !important;
    color: var(--muted); border-radius: 10px 10px 0 0 !important;
    padding: 10px 14px !important;
}
.gradio-container button[role="tab"]:hover { color: var(--ink); }
.gradio-container button[role="tab"].selected,
.gradio-container button[role="tab"][aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom: 3px solid var(--indigo) !important;
}

/* ---------- Composants Gradio ---------- */
.gradio-container button.primary {
    background: var(--indigo) !important; border: none !important;
    font-weight: 600 !important; color: #fff !important;
}
.gradio-container button.primary:hover { background: var(--indigo-deep) !important; }
.gradio-container button.secondary { font-weight: 600 !important; }
.gradio-container .block { border-color: var(--line); }
.gradio-container .gallery-item { border-radius: 10px !important; overflow: hidden; }
.gradio-container table thead th {
    background: color-mix(in srgb, var(--indigo) 8%, var(--panel)) !important;
    font-weight: 600;
}
.gradio-container code {
    background: color-mix(in srgb, var(--indigo) 10%, transparent);
    color: var(--indigo-deep);
    padding: 2px 6px; border-radius: 5px; font-size: .9em;
}
.dark .gradio-container code { color: #C9CCFF; }
.gradio-container pre {
    background: #12142B !important; color: #F3F4FF !important;
    border-radius: 10px; padding: 12px !important;
}
.gradio-container pre code { background: transparent; color: inherit; }

/* Accessibilité : focus clavier visible */
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
    outline: 3px solid color-mix(in srgb, var(--indigo) 55%, white) !important;
    outline-offset: 2px;
}

/* ---------- Footer ---------- */
.app-footer {
    margin-top: 36px; padding: 28px 30px; border-radius: 18px;
    background: var(--panel); border: 1px solid var(--line);
    color: var(--muted); font-size: .92rem;
}
.app-footer h3 { color: var(--ink) !important; margin-top: 0; }
.app-footer .about { max-width: 78ch; line-height: 1.6; color: var(--ink); }
.app-footer .mcard { background: var(--paper); }
.app-footer .stack { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px 0; }
.app-footer .stack .chip { border-color: var(--line); color: var(--ink); }
.app-footer .done { margin: 16px 0 0 0; font-size: .85rem; color: var(--muted); }

/* ---------- Responsive ---------- */
@media (max-width: 960px) {
    .hero { grid-template-columns: 1fr; padding: 26px 22px; }
    #sidebar { position: static !important; width: 100% !important; max-width: 100% !important; }
    #content-col { margin-left: 0 !important; }
    #nav-radio .wrap { flex-direction: row !important; flex-wrap: wrap !important; }
    #nav-radio label { width: auto; padding: 8px 12px !important; }
}

/* ---------- Mouvement réduit ---------- */
@media (prefers-reduced-motion: reduce) {
    .cstl circle, .cstl text { animation: none !important; opacity: 1 !important; }
    .kpi, .mcard, .qcard, #nav-radio label { transition: none !important; }
}
"""


# =============================================================================
# BLOCS HTML
# =============================================================================
def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _constellation(classes: Iterable[str]) -> str:
    """Illustration SVG : dix classes qui se regroupent dans l'espace d'embedding."""
    classes = list(classes)
    n = max(len(classes), 1)
    rnd = random.Random(7)  # déterministe : même rendu à chaque lancement
    W, H = 640, 380
    cx0, cy0, rx, ry = W / 2, H / 2, 232, 128

    dots: List[str] = []
    labels: List[str] = []
    for i, name in enumerate(classes):
        ang = 2 * math.pi * i / n - math.pi / 2
        cx = cx0 + rx * math.cos(ang)
        cy = cy0 + ry * math.sin(ang)
        col = CLASS_PALETTE[i % len(CLASS_PALETTE)]
        for _ in range(16):
            ex = cx + rnd.gauss(0, 15)
            ey = cy + rnd.gauss(0, 11)
            sx = rnd.uniform(30, W - 30) - ex
            sy = rnd.uniform(24, H - 24) - ey
            delay = rnd.uniform(0, 0.7)
            dots.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.3" fill="{col}" '
                f'style="--sx:{sx:.0f}px;--sy:{sy:.0f}px;animation-delay:{delay:.2f}s"/>'
            )
        ly = cy - 26 if math.sin(ang) < 0.25 else cy + 36
        labels.append(
            f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle">{_esc(str(name).lower())}</text>'
        )

    ring = f'<ellipse cx="{cx0}" cy="{cy0}" rx="{rx}" ry="{ry}"/>'
    return (
        f'<svg class="cstl" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Illustration : les embeddings se regroupent par classe">'
        f'{ring}{"".join(dots)}{"".join(labels)}</svg>'
    )


def hero_html(
    classes: Iterable[str],
    model_names: Iterable[str],
    n_tabs: int,
    n_images: int,
    n_tests: int = 53,
) -> str:
    model_names = list(model_names)
    legend = "".join(
        f'<span class="chip" style="--c:{MODEL_COLORS.get(m, "#8A8FD8")}"><i></i>{_esc(m)}</span>'
        for m in model_names
    )
    badges = (
        f'<span class="status-badge badge-success">✅ {n_tabs} sections</span>'
        f'<span class="status-badge badge-info">🧪 {n_tests} tests</span>'
        f'<span class="status-badge badge-info">🧠 {len(model_names)} modèles</span>'
        f'<span class="status-badge badge-warning">📊 {n_images} images indexées</span>'
    )
    return f"""
    <section class="hero">
        <div>
            <div class="brand"><b>🎨</b> CLIP LoRA Studio</div>
            <h1>Ce que LoRA change dans CLIP, et ce qu'il préserve.</h1>
            <p class="lead"><b>Explorez, comparez, comprenez</b> comment LoRA fine-tune CLIP
            sur Fashion-MNIST : quatre modèles, dix classes, et une question — jusqu'où adapter
            sans perdre la généralité ?</p>
            <div class="badges">{badges}</div>
            <div class="legend">{legend}</div>
        </div>
        <figure>
            {_constellation(classes)}
            <figcaption>Illustration : dix classes qui se regroupent dans l'espace d'embedding</figcaption>
        </figure>
    </section>
    """


def kpi_html(cards) -> str:
    """Grille de cartes KPI, responsive (auto-fit)."""
    cells = []
    for c in cards:
        color = c.get("color", "#4B4FE0")
        value = _esc(c.get("value", "—"))
        label = _esc(c.get("label", ""))
        sub = _esc(c.get("sub", ""))
        cells.append(
            f'<div class="kpi" style="--c:{color}">'
            f'<div class="v">{value}</div>'
            f'<div class="l">{label}</div>'
            f'<div class="s">{sub}</div>'
            f"</div>"
        )
    n = len(cards)
    return f'<div class="kpis" data-count="{n}">{"".join(cells)}</div>'


def status_html(model_names: Iterable[str], retriever_ok: bool, n_images: int) -> str:
    chips = "".join(
        f'<span class="chip" style="--c:{MODEL_COLORS.get(m, "#8A8FD8")}"><i></i>{_esc(m)}</span>'
        for m in model_names
    )
    if retriever_ok:
        ret = f'<span class="ok">🔍 {n_images} images indexées</span>'
    else:
        ret = '<span class="bad">⚠️ Retriever non disponible</span>'
    return (
        '<div class="statusbar">'
        f'<span class="grp"><b>Modèles chargés</b> {chips}</span>'
        f'<span class="grp"><b>Retriever</b> {ret}</span>'
        "</div>"
    )


def demo_note_html(tag: str, text: str, color: str = "#4B4FE0") -> str:
    return (
        f'<div class="demo-note" style="--c:{_esc(color)}">'
        f'<span class="tag">{_esc(tag)}</span><span>{_esc(text)}</span></div>'
    )


def sidebar_brand_html() -> str:
    """Libellé du menu latéral (la marque est dans le header fixe)."""
    return '<div class="side-label">Navigation</div>'


def topbar_html(section_label: str = "🏠 Accueil") -> str:
    """Header fixe : marque + section courante."""
    return (
        '<div class="topbar-in">'
        '<div class="logo">🎨</div>'
        '<div><div class="t">CLIP LoRA Studio</div>'
        '<div class="s">Fashion-MNIST · 4 modèles</div></div>'
        f'<div class="crumb">{_esc(section_label)}</div>'
        "</div>"
    )


def section_html(icon: str, title: str, sub: str = "") -> str:
    """Titre de section de la page d'accueil."""
    sub_html = f'<span class="sub">{_esc(sub)}</span>' if sub else ""
    return (
        f'<div class="sec"><span class="ic">{_esc(icon)}</span>'
        f"<h3>{_esc(title)}</h3>{sub_html}</div>"
    )


def model_cards_html() -> str:
    models = [
        ("Zero-shot", "CLIP baseline (aucun fine-tuning)", MODEL_COLORS["Zero-shot"]),
        ("A1 LinearHead", "5K paramètres, entraînement minimal", MODEL_COLORS["A1 LinearHead"]),
        ("A2 LoRA V+T", "LoRA sur vision + texte (96 couches)", MODEL_COLORS["A2 LoRA V+T"]),
        ("B LoRA V only", "LoRA sur vision uniquement (48 couches, 2× plus léger)", MODEL_COLORS["B LoRA V only"]),
    ]
    cards = "".join(
        f'<div class="mcard" style="--c:{c}"><b>{_esc(n)}</b><span>{_esc(d)}</span></div>'
        for n, d, c in models
    )
    return f'<div class="models">{cards}</div>'


def insights_html(delta_line: str = "") -> str:
    """Cartes d'insights de la page d'accueil."""
    items = [
        ("📈", "Gain du fine-tuning", delta_line or "LoRA V+T améliore nettement l'accuracy vs Zero-shot."),
        ("⚖️", "Meilleur compromis", "LoRA V only : 2× plus léger pour environ 98 % des performances d'A2."),
        ("⚡", "Budget ultra-réduit", "A1 LinearHead : efficace avec seulement 5K paramètres."),
        ("🧠", "Généralité préservée", "Le drift angulaire mesure combien l'encodeur a été modifié."),
    ]
    cells = "".join(
        f'<div class="insight"><div class="ic">{ic}</div>'
        f"<div><b>{_esc(t)}</b><span>{_esc(d)}</span></div></div>"
        for ic, t, d in items
    )
    return f'<div class="insights">{cells}</div>'


def footer_html(n_tabs: int, n_tests: int = 53) -> str:
    stack = "".join(
        f'<span class="chip">{_esc(s)}</span>'
        for s in ["Python", "PyTorch", "CLIP (ViT-B/32)", "Gradio 4+", "Plotly", "LoRA"]
    )
    return f"""
    <div class="app-footer">
        <h3>📚 À propos</h3>
        <p class="about">Projet académique étudiant le fine-tuning de <b>CLIP</b> avec <b>LoRA</b>
        sur <b>Fashion-MNIST</b>. Étude du trade-off <b>adaptation ↔ préservation de la généralité</b>.</p>

        <p><b>🧠 Modèles comparés</b></p>
        {model_cards_html()}

        <p><b>🛠️ Stack technique</b></p>
        <div class="stack">{stack}</div>

        <p><b>📊 Fonctionnalités :</b> {n_tabs} sections · {n_tests} tests automatisés · Export CSV/Markdown</p>

        <p class="done">✅ Phase 4 complète — Prêt pour soutenance · © 2026</p>
    </div>
    """