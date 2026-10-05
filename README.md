🎨 CLIP LoRA Studio

Étude interactive du fine-tuning de CLIP avec LoRA sur Fashion-MNIST —
explorez, comparez et comprenez le trade-off adaptation ↔ préservation de la généralité.








📸 Aperçu

CLIP LoRA Studio est une application Gradio qui compare 4 modèles sur
Fashion-MNIST à travers 11 sections interactives : classification, retrieval,
cross-modal, drift, calibration, et plus.

┌──────────────────────────────────────────────────────────────┐
│ 🏠 Accueil · 🎯 Classification · 🎨 Comparateur · 🔤 Prompt Lab │
│ 🔍 Retrieval · 🔎 Cross-Modal · 🧭 Drift · 📊 Analytics │
│ 🌡️ Calibration · 🧪 Playground · 📥 Exports · 🧠 Insights │
└──────────────────────────────────────────────────────────────┘

Chiffres clés du projet :

Métrique

Valeur

Modèle

🎯 Accuracy max

0.874

A2 LoRA V+T

📈 Gain vs Zero-shot

+37.4 pts

A2 LoRA V+T

🔍 Retrieval P@1

0.900

A2 LoRA V+T

⚖️ Meilleur compromis

B LoRA V only

0.863 acc · 491K params

🧪 Tests automatisés

53

tous passent ✅

📁 Images indexées

2 000

Fashion-MNIST

✨ Fonctionnalités

🎯 Classification

Upload d'une image → prédiction Top-5 + distribution des probabilités + tableau complet.
Sélection du modèle et du Top-K interactifs.

🎨 Comparateur Visuel

Les 4 modèles côte à côte sur la même image, avec indicateur de consensus
(🟢 unanime · 🟡 partiel · 🔴 désaccord) et export CSV.

🔤 Prompt Lab

Testez des templates de prompts personnalisés (a photo of a {},
a black and white photo of a {}, etc.) et observez l'effet sur la prédiction.

🔍 Retrieval (Image → Image)

Trouvez les k images les plus similaires dans 2 000 images Fashion-MNIST.
Filtre par classe, choix de la métrique (cosinus / euclidienne / dot product).

🔎 Cross-Modal (Texte → Image)

Tapez "a black sneaker" ou "a red dress" → retrouvez les images correspondantes
via l'alignement CLIP texte/image.

🧭 Drift Explorer

Analysez l'impact du fine-tuning classe par classe : quelle classe gagne le
plus, quelle classe perd, quel modèle dérive le moins.

📊 Analytics

Dashboard récapitulatif : tableau TEAM_FINAL_TABLE, heatmap, graphe de Pareto
(trade-off accuracy ↔ retrieval ↔ params).

🌡️ Calibration

Reliability diagram, distribution des confiances, simulateur interactif (ECE).

🧪 Playground Multi-Images

Uploadez plusieurs images d'un coup → prédictions batch, consensus, export CSV.

📥 Exports

Centre de téléchargement de tous les CSV / Markdown générés par l'app.

🧠 Insights

Robustesse au bruit, oubli catastrophique (OOD), multi-seed, budget de paramètres.

📖 Guide de démo

Parcours de 5 minutes, questions probables du jury, checklist de soutenance.

🚀 Installation rapide

Prérequis

Python 3.10+

pip

(Optionnel) GPU CUDA pour accélérer l'inférence

~2 GB d'espace disque (checkpoints + assets)

Setup

# 1. Clone du projet
git clone <ton-repo> clip-lora-studio
cd clip-lora-studio

# 2. Environnement virtuel
python -m venv Clip-lora-studio-env

# Windows
Clip-lora-studio-env\Scripts\activate

# Linux / macOS
source Clip-lora-studio-env/bin/activate

# 3. Dépendances
pip install -r requirements.txt

Préparation des données

# Générer les images d'éval + embeddings CLIP
python scripts/prepare_eval_images.py

# Régénérer les labels (si nécessaire)
python scripts/regenerate_labels.py

Lancement

python app.py

L'application démarre sur http://localhost:7860.

📁 Structure du projet

clip-lora-studio/
├── app.py                       # Application Gradio (11 sections)
├── ui_theme.py                  # Thème, CSS, hero, KPI, footer
├── requirements.txt             # Dépendances Python
├── requirements-dev.txt         # Dépendances de test
├── pytest.ini                   # Config pytest
├── README.md                    # Ce fichier
├── CHANGELOG.md                 # Historique des versions
├── RAPPORT_AMELIORATION.md      # Bugs corrigés + recommandations
├── INSTALL.md                   # Guide d'installation détaillé
│
├── modules/                     # Logique métier
│   ├── models.py                # ModelRegistry (charge les 4 modèles CLIP+LoRA)
│   ├── inference.py             # classify, get_image_embedding, get_text_embeddings
│   ├── retrieval.py             # ImageRetriever multi-modèle
│   └── analytics.py             # Analytics (team_table, figures, CSV)
│
├── utils/
│   └── constants.py             # CLASSES, chemins, device
│
├── scripts/
│   ├── prepare_eval_images.py   # Prépare les 2000 images + embeddings
│   ├── regenerate_labels.py     # Régénère eval_labels.npy
│   └── ...
│
├── tests/                       # 53 tests automatisés
│   ├── conftest.py              # Fixtures
│   ├── test_retrieval.py        # 20 tests
│   ├── test_classification.py   # 11 tests
│   ├── test_analytics.py        # 12 tests
│   ├── test_consistency.py      # 5 tests
│   └── test_app_smoke.py        # 5 tests
│
├── checkpoints/                 # Modèles fine-tunés (LoRA)
│   ├── A1_linearhead/
│   ├── A2_lora/
│   └── B_lora/
│
├── assets/
│   ├── data/                    # CSV + embeddings .npy
│   ├── eval_images/             # 2000 PNG indexés
│   ├── examples/                # Images d'exemple pour l'UI
│   └── figures/team/            # 22 figures générées
│
├── exports/                     # Sorties CSV / Markdown (créé au runtime)
└── logs/                        # app.log rotatif (créé au runtime)

🎯 Utilisation

Scénario 1 — Comparer les modèles sur une image

Ouvrir 🏠 Accueil

Cliquer sur 🎨 Comparateur dans le menu latéral (ou "Accès rapide")

Upload une image (une sneaker par exemple)

Cliquer sur 🔬 Comparer les 4 modèles

Observer : consensus, prédictions, tableau récap, export CSV

Scénario 2 — Tester un prompt

🔤 Prompt Lab

Upload une image

Modifier le template : a black and white photo of a {}

🚀 Tester ce template

Observer la nouvelle prédiction

Scénario 3 — Recherche cross-modale

🔎 Cross-Modal

Taper a red dress

🔍 Rechercher → galerie des images les plus proches

Scénario 4 — Analyse batch

🧪 Playground

Upload 5-10 images d'un coup

Choisir "Tous les modèles"

🚀 Analyser le batch → tableau + export CSV

🧠 Modèles comparés

Modèle

Params

Accuracy

Retrieval P@1

Drift

Description

Zero-shot

0

0.500

0.400

0.000

CLIP baseline (aucun fine-tuning)

A1 LinearHead

5 130

0.670

0.400

0.000

Head de classification uniquement

A2 LoRA V+T

983 040

0.874

0.900

0.240

LoRA sur vision + texte (96 couches)

B LoRA V only

491 520

0.863

0.800

0.232

LoRA vision uniquement (48 couches)

Variantes testées : A2 λ=0.25 / 0.5 / 1.0 (ablation du poids de régularisation).

📊 Résultats clés

Gain du fine-tuning

Zero-shot       ████████                     0.500
A1 LinearHead   ███████████                  0.670   +17.0 pts
B LoRA V only   ██████████████               0.863   +36.2 pts
A2 LoRA V+T     ██████████████▌              0.874   +37.4 pts  ⭐

Trade-off params ↔ performance

A1 : ultra-léger (5K params) → efficacité minimale acceptable

B : équilibre (491K params) → meilleur compromis

A2 : performance max (983K params) → drift plus élevé

🔬 Détails techniques

Architecture CLIP

Backbone : openai/clip-vit-base-patch32 (ViT-B/32)

Dimension embedding : 512

Image size : 224×224

Text context : 77 tokens

Implémentation LoRA

Rank : r=8 (LoRA complet) / r=1 (LoRA minimal, ablation)

Target modules : Q, K, V, O (attention) + MLP

Alpha : 1.0 (par défaut) ou variantes 0.25 / 0.5 / 1.0 (ablation)

Injection : manuelle (pas de PEFT library) → transparence totale

Entraînement

Optimizer : AdamW

Learning rate : 1e-4 (tête) / 5e-5 (LoRA)

Batch size : 64

Epochs : 5

Seeds testés : 42, 123, 2024

Évaluation

Classification accuracy : Top-1 sur 2000 images

Retrieval P@1 : text→image precision@1

Drift angulaire : distance cosine moyenne entre embeddings avant/après

ECE : Expected Calibration Error (15 bins)

🧪 Tests

# Tous les tests
pytest tests/ -v

# Un fichier
pytest tests/test_retrieval.py -v

# Avec rapport HTML
pytest tests/ --html=reports/test_report.html --self-contained-html

# Uniquement les rapides
pytest tests/ -v -m "not slow"

Résultat attendu :

==================== 53 passed in ~40s ====================

Couverture :

Fichier

Tests

Couverture

test_retrieval.py

20

Retriever, multi-modèle, métriques, accuracy@5

test_classification.py

11

classify, classify_multi, embeddings

test_analytics.py

12

team_table, figures, CSV, calibration

test_consistency.py

5

labels ↔ images ↔ embeddings

test_app_smoke.py

5

Imports, build_app, chemins

🐛 Bugs connus & limitations

Bug / Limitation

Statut

Contournement

Seul clip_image_zs.npy est généré

✅ Attendu

Lancer prepare_eval_images.py --model a2_lora_vt

Démarrage ~30s (chargement 4 modèles CLIP)

✅ Normal

Une seule fois par session

Pas d'inférence GPU par défaut sur CPU

✅ Normal

DEVICE = "cuda" si GPU dispo

🛠️ Scripts utilitaires

Script

Rôle

scripts/prepare_eval_images.py

Génère 2000 images + embeddings CLIP

scripts/regenerate_labels.py

Régénère eval_labels.npy

scripts/audit.py

Vérifie la cohérence des assets

scripts/run_tests.ps1

Lance tous les tests + rapport HTML

🎓 Contexte académique

Ce projet étudie la question suivante :

Jusqu'où peut-on adapter CLIP à un domaine spécifique (Fashion-MNIST)
sans perdre sa généralité cross-modale ?

Réponse apportée :

LoRA V+T offre +37.4 pts d'accuracy vs Zero-shot

LoRA V only offre 98% des performances de A2 avec 50% des paramètres

Le drift angulaire mesure objectivement l'adaptation de l'encoder

Un bon compromis existe : B LoRA V only maximise perf / budget

📚 Références

CLIP : Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021

LoRA : Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022

Fashion-MNIST : Xiao et al., Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017

Gradio : Abid et al., Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild, 2019

👥 Auteurs

Wael G. — Développement, entraînement, UI

Projet encadré dans le cadre d'un cursus académique

📄 Licence

MIT — voir LICENSE pour détails.


<p align="center">
  <b>✅ Phase 4 complète — 11 sections, 53 tests, 4 modèles</b><br>
  <i>Prêt pour soutenance · © 2026</i>
</p>