# 📝 Changelog — CLIP LoRA Studio

Toutes les modifications notables du projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/) et
[Semantic Versioning](https://semver.org/).

---

## [4.6.0] — 2026-10-05 — Documentation finale

### Ajouté
- 📄 `README.md` complet (badges, structure, exemples, installation)
- 📋 `RAPPORT_AMELIORATION.md` (bugs corrigés + recommandations)
- 📝 `CHANGELOG.md` (ce fichier)
- 🔧 `INSTALL.md` (guide d'installation pas à pas)

### Modifié
- Footer avec nouvelles métriques (11 sections, 53 tests)
- Onglet Guide : ajout de la checklist soutenance

---

## [4.5.0] — 2026-10-04 — Exports & Polish

### Ajouté
- 📥 **Section Exports** : centre de téléchargement des CSV / Markdown
- 📥 Bouton d'export dans **Comparateur** et **Playground**
- 🎨 **Header pro** avec gradient violet + badges de statut
- 🎨 **Footer** professionnel avec cartes modèles
- 🎨 **Page d'accueil** (hero animé, KPIs, accès rapides, graphiques)
- 🎨 **Menu latéral** sticky et scrollable
- 🌓 **Mode clair/sombre** toggle
- 📖 **Section Guide** : parcours 5 min + questions jury + checklist

### Modifié
- `ui_theme.py` : refonte complète du CSS (tokens, responsive)
- `app.py` : `build_app()` avec sidebar + tabs masqués
- KPI : filtrage sur les 4 modèles canoniques
- 6 KPIs intelligents (accuracy, gain, compromis, P@1, tests, images)

---

## [4.4.0] — 2026-10-03 — Cross-Modal Retrieval

### Ajouté
- 🔎 **Section Cross-Modal** : recherche texte → images
- Utilise l'encodeur texte de CLIP
- 8 exemples cliquables ("a black sneaker", "a red dress"...)
- ✅ 2 tests : `test_cross_modal_text_to_image`, `test_cross_modal_scores_in_range`

---

## [4.3.0] — 2026-10-02 — Playground Multi-Images

### Ajouté
- 🧪 **Section Playground** : upload batch (drag & drop multiple)
- Prédictions sur tous les modèles en parallèle
- Statistiques agrégées par modèle
- Export CSV automatique
- ✅ 3 tests : dataframe structure, CSV export, batch mock

---

## [4.2.0] — 2026-10-01 — Calibration

### Ajouté
- 🌡️ **Section Calibration** : ECE, reliability diagram, distribution
- Simulateur interactif (verdict automatique)
- ✅ 2 tests : calibration figures + probability confidence

---

## [4.1.0] — 2026-09-30 — Comparateur Visuel

### Ajouté
- 🎨 **Section Comparateur** : grille 2×2 des 4 modèles
- Indicateur de consensus (🟢 / 🟡 / 🔴)
- Tableau récapitulatif + export CSV
- ✅ 2 tests : `classify_multi` structure + consensus

---

## [4.0.0] — 2026-09-29 — Refactoring majeur

### Ajouté
- 📊 Bloc diagnostic au démarrage
- 📝 Logging structuré avec rotation
- 🎯 KPI dynamiques dans l'UI
- 🔧 Helpers d'export (`_export_dataframe`, `_export_markdown_summary`)
- ⚙️ Diagnostic au démarrage

### Modifié
- Réécriture complète de `modules/retrieval.py`
- Refonte de `build_app()` avec diagnostic
- Migration vers `logger` (fin des `print`)

### Corrigé
- 🐛 **Bug 3** : erreurs silencieuses (`try/except pass`) → logging
- 🐛 **Bug 2** : figures manquantes dans Drift Explorer
- 🐛 **Bug 1** : retrieval incohérent (labels désalignés)

---

## [3.5.0] — 2026-09-28 — Fix retrieval critique

### Corrigé
- 🐛 **`eval_labels.npy` régénéré** : 2000 labels, ~200/classe
- 🐛 **Normalisation L2** au chargement et à l'encodage
- 🐛 **Filtre par modèle** : dropdown ne propose que les embeddings dispo
- 🐛 **Accuracy@5 = 88%** validé

### Ajouté
- ✅ Tests `test_retrieval.py` : 20 tests
- ✅ Test `test_accuracy_at_5` (seuil 80%)
- ✅ Test `test_labels_balanced` (tolérance 150-250)

---

## [3.0.0] — 2026-09-27 — Suite de tests

### Ajouté
- 🧪 **53 tests automatisés** (unitaires + intégration + smoke)
- `pytest.ini` avec marqueurs (`slow`, `gpu`)
- `requirements-dev.txt` (pytest, pytest-html, pytest-cov)
- Fixtures partagées dans `conftest.py`

### Modifié
- `classify(None)` lève maintenant `ValueError` (au lieu de retourner None)

---

## [2.5.0] — 2026-09-26 — Multi-modèle retrieval

### Ajouté
- 🔧 `DEFAULT_EMBEDDINGS_MAP` dans `retrieval.py`
- Support de 4 embeddings (`zs`, `a1`, `a2`, `b`)
- `available_models` property
- `get_unavailable_models()` avec hints actionnables

### Modifié
- `ImageRetriever.__init__` accepte un dict au lieu d'un chemin unique
- Filtre par classe dans `retrieve()`
- Métrique configurable (cosine / euclidean / dot)

---

## [2.0.0] — 2026-09-25 — Drift Explorer

### Ajouté
- 🧭 **Section Drift Explorer** : impact du fine-tuning par classe
- Visualisation interactive (dropdown métrique + tri)
- Graphique comparatif ZS vs A2 vs B

---

## [1.5.0] — 2026-09-24 — Analytics

### Ajouté
- 📊 **Section Analytics** : dashboard récapitulatif
- Tableau TEAM_FINAL_TABLE
- Heatmap + graphe de Pareto
- Insights clés

---

## [1.0.0] — 2026-09-23 — Version initiale

### Ajouté
- 🎯 **Section Classification** : upload + top-K + distribution
- 🔤 **Section Prompt Lab** : templates personnalisés
- 🔍 **Section Retrieval** : recherche image → image
- 🧠 **Section Insights** : robustesse, OOD, multi-seed
- App Gradio avec 6 onglets natifs

### Modèles entraînés
- Zero-shot (baseline)
- A1 LinearHead (5K params)
- A2 LoRA V+T (983K params)
- B LoRA V only (491K params)

---

## 🏷️ Légende des versions

- **X.0.0** : nouvelle section majeure ou refonte
- **X.Y.0** : nouvelle fonctionnalité
- **X.Y.Z** : correction de bug ou amélioration mineure

---

<p align="center">
  <i>Historique complet du projet · © 2026</i>
</p>