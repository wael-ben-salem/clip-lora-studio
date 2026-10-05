# 🎨 CLIP LoRA Studio

> **Étude interactive du fine-tuning de CLIP avec LoRA sur Fashion-MNIST**
>
> Explorez, comparez et comprenez le compromis entre **adaptation à une tâche spécifique** et **préservation des capacités générales cross-modales** de CLIP.

<p align="center">

**Classification · Retrieval · Cross-Modal · Drift · Calibration · Analytics · LoRA**

</p>

---

## 📸 Aperçu

**CLIP LoRA Studio** est une application interactive développée avec **Gradio** pour étudier expérimentalement l'impact du fine-tuning de **CLIP** avec **LoRA** sur le dataset **Fashion-MNIST**.

L'application permet de comparer **4 configurations de modèles** à travers **11 sections interactives**, couvrant notamment :

* classification d'images ;
* comparaison multi-modèles ;
* prompt engineering ;
* image retrieval ;
* recherche cross-modale texte → image ;
* analyse du drift des représentations ;
* analytics et visualisations ;
* calibration des probabilités ;
* analyse batch ;
* exports ;
* insights expérimentaux.

### 🧭 Navigation

```text
🏠 Accueil
 ├── 🎯 Classification
 ├── 🎨 Comparateur
 ├── 🔤 Prompt Lab
 ├── 🔍 Retrieval
 ├── 🔎 Cross-Modal
 ├── 🧭 Drift
 ├── 📊 Analytics
 ├── 🌡️ Calibration
 ├── 🧪 Playground
 ├── 📥 Exports
 └── 🧠 Insights
```

---

## 📊 Chiffres clés

| Indicateur            |                     Résultat | Configuration |
| --------------------- | ---------------------------: | ------------- |
| 🎯 Accuracy maximale  |                    **0.874** | A2 LoRA V+T   |
| 📈 Gain vs Zero-shot  |                **+37.4 pts** | A2 LoRA V+T   |
| 🔍 Retrieval P@1      |                    **0.900** | A2 LoRA V+T   |
| ⚖️ Meilleur compromis | **0.863 acc. / 491K params** | B LoRA V only |
| 🧪 Tests automatisés  |           **53 / 53 passés** | Tous          |
| 📁 Images indexées    |                    **2 000** | Fashion-MNIST |

---

# ✨ Fonctionnalités

## 🎯 Classification

Uploadez une image et obtenez :

* la prédiction Top-K ;
* la distribution des probabilités ;
* les scores de toutes les classes ;
* le choix du modèle ;
* le choix du nombre de prédictions affichées.

---

## 🎨 Comparateur Visuel

Comparez les **4 modèles sur exactement la même image**.

L'interface affiche :

* la prédiction de chaque modèle ;
* les probabilités ;
* le consensus entre modèles ;
* un indicateur visuel :

  * 🟢 **unanime**
  * 🟡 **partiel**
  * 🔴 **désaccord**
* un tableau comparatif ;
* un export CSV.

Cette section permet d'observer directement comment le fine-tuning modifie le comportement de CLIP.

---

## 🔤 Prompt Lab

Testez différents templates de prompts et observez leur influence sur la classification.

Exemples :

```text
a photo of a {}
a black and white photo of a {}
a fashion item called a {}
a grayscale image of a {}
```

Le même modèle peut produire des prédictions différentes selon la formulation du prompt.

---

## 🔍 Retrieval — Image → Image

Recherchez les images les plus similaires à une image donnée dans un index de **2 000 images Fashion-MNIST**.

Fonctionnalités :

* choix du modèle ;
* choix du nombre de résultats ;
* filtrage par classe ;
* métrique cosine ;
* distance euclidienne ;
* dot product ;
* visualisation des images retrouvées.

---

## 🔎 Cross-Modal — Texte → Image

Explorez l'alignement texte-image de CLIP.

Exemples de requêtes :

```text
a black sneaker
a red dress
a white shirt
a pair of trousers
```

Le texte est encodé avec l'encodeur textuel de CLIP puis comparé aux embeddings des images.

Cette fonctionnalité permet d'évaluer directement la capacité **cross-modale** du modèle.

---

## 🧭 Drift Explorer

Analysez l'impact du fine-tuning sur les représentations internes.

L'explorateur permet notamment d'observer :

* les classes qui progressent le plus ;
* les classes qui régressent ;
* le drift par classe ;
* le modèle qui préserve le mieux ses représentations ;
* l'effet de l'adaptation sur les embeddings.

---

## 📊 Analytics

Dashboard expérimental regroupant :

* `TEAM_FINAL_TABLE` ;
* heatmaps ;
* courbes comparatives ;
* métriques de classification ;
* retrieval ;
* paramètres entraînables ;
* drift ;
* graphe de Pareto.

L'objectif est de visualiser le **trade-off performance ↔ généralité ↔ budget de paramètres**.

---

## 🌡️ Calibration

Analysez la fiabilité des probabilités produites par les modèles.

Fonctionnalités :

* reliability diagram ;
* distribution des niveaux de confiance ;
* Expected Calibration Error (**ECE**) ;
* simulateur interactif ;
* comparaison entre modèles.

---

## 🧪 Playground Multi-Images

Uploadez plusieurs images simultanément afin d'effectuer une analyse batch.

Le playground permet de :

* sélectionner un ou plusieurs modèles ;
* prédire plusieurs images ;
* comparer les résultats ;
* calculer le consensus ;
* exporter les résultats au format CSV.

---

## 📥 Exports

Un centre d'export permet de récupérer les résultats générés par l'application :

* CSV ;
* Markdown ;
* tableaux analytiques ;
* résultats de comparaison ;
* résultats batch.

Les fichiers sont générés dans :

```text
exports/
```

---

## 🧠 Insights

Une section dédiée à l'interprétation expérimentale des résultats.

Elle couvre notamment :

* robustesse au bruit ;
* oubli catastrophique ;
* comportement OOD ;
* analyse multi-seed ;
* budget de paramètres ;
* compromis adaptation / préservation.

---

## 📖 Guide de démo

Une section dédiée à la présentation du projet permet de préparer une démonstration de quelques minutes.

Elle contient notamment :

* un parcours de démonstration de **5 minutes** ;
* les questions probables du jury ;
* les résultats importants à présenter ;
* une checklist de soutenance.

---

# 🚀 Installation

## Prérequis

* **Python 3.10+**
* `pip`
* Git
* environ **2 GB d'espace disque**
* GPU CUDA recommandé mais optionnel

> L'application peut fonctionner sur CPU. Un GPU permet cependant d'accélérer le chargement et l'inférence des modèles.

---

## 1. Cloner le projet

```bash
git clone <TON_REPOSITORY_URL>
cd clip-lora-studio
```

---

## 2. Créer l'environnement virtuel

### Windows

```powershell
python -m venv Clip-lora-studio-env

Clip-lora-studio-env\Scripts\activate
```

### Linux / macOS

```bash
python -m venv Clip-lora-studio-env
source Clip-lora-studio-env/bin/activate
```

---

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Pour installer également les dépendances de développement :

```bash
pip install -r requirements-dev.txt
```

---

# 📦 Préparation des données

Avant la première utilisation, préparer les images d'évaluation et les embeddings.

```bash
python scripts/prepare_eval_images.py
```

Si les labels doivent être régénérés :

```bash
python scripts/regenerate_labels.py
```

### Assets principaux générés

```text
assets/
├── data/
│   ├── *.csv
│   └── *.npy
│
└── eval_images/
    └── 2000 images Fashion-MNIST
```

---

# ▶️ Lancement

Lancer l'application avec :

```bash
python app.py
```

L'interface Gradio est ensuite accessible localement à :

```text
http://localhost:7860
```

---

# 📁 Structure du projet

```text
clip-lora-studio/
│
├── app.py
│   └── Application Gradio principale
│
├── ui_theme.py
│   └── Thème, CSS, hero, KPI et footer
│
├── requirements.txt
│   └── Dépendances runtime
│
├── requirements-dev.txt
│   └── Dépendances de développement et tests
│
├── pytest.ini
│   └── Configuration pytest
│
├── README.md
│   └── Documentation du projet
│
├── CHANGELOG.md
│   └── Historique des versions
│
├── RAPPORT_AMELIORATION.md
│   └── Bugs corrigés et recommandations
│
├── INSTALL.md
│   └── Guide d'installation détaillé
│
├── modules/
│   │
│   ├── models.py
│   │   └── ModelRegistry et chargement des modèles
│   │
│   ├── inference.py
│   │   └── Classification et génération d'embeddings
│   │
│   ├── retrieval.py
│   │   └── ImageRetriever multi-modèle
│   │
│   └── analytics.py
│       └── Analytics, figures et exports CSV
│
├── utils/
│   └── constants.py
│       └── Classes, chemins et configuration du device
│
├── scripts/
│   ├── prepare_eval_images.py
│   │   └── Préparation des 2000 images et embeddings
│   │
│   ├── regenerate_labels.py
│   │   └── Régénération des labels
│   │
│   ├── audit.py
│   │   └── Audit de cohérence des assets
│   │
│   └── ...
│
├── tests/
│   ├── conftest.py
│   ├── test_retrieval.py
│   ├── test_classification.py
│   ├── test_analytics.py
│   ├── test_consistency.py
│   └── test_app_smoke.py
│
├── checkpoints/
│   ├── A1_linearhead/
│   ├── A2_lora/
│   └── B_lora/
│
├── assets/
│   ├── data/
│   ├── eval_images/
│   ├── examples/
│   └── figures/
│       └── team/
│
├── exports/
│   └── Généré au runtime
│
└── logs/
    └── app.log généré au runtime
```

---

# 🎯 Guide d'utilisation

## Scénario 1 — Comparer les modèles

1. Ouvrir **🏠 Accueil**
2. Aller dans **🎨 Comparateur**
3. Uploader une image
4. Cliquer sur **Comparer les 4 modèles**
5. Observer :

   * les prédictions ;
   * le consensus ;
   * les scores ;
   * le tableau comparatif.
6. Exporter les résultats en CSV.

---

## Scénario 2 — Tester un prompt

1. Ouvrir **🔤 Prompt Lab**
2. Uploader une image
3. Modifier le template :

```text
a black and white photo of a {}
```

4. Cliquer sur **Tester ce template**
5. Comparer la nouvelle prédiction.

---

## Scénario 3 — Recherche cross-modale

1. Ouvrir **🔎 Cross-Modal**
2. Entrer une requête :

```text
a red dress
```

3. Cliquer sur **Rechercher**
4. Observer les images correspondant le mieux au texte.

---

## Scénario 4 — Analyse batch

1. Ouvrir **🧪 Playground**
2. Uploader 5 à 10 images
3. Sélectionner **Tous les modèles**
4. Cliquer sur **Analyser le batch**
5. Examiner le tableau et le consensus
6. Exporter les résultats.

---

# 🧠 Modèles comparés

| Modèle            | Paramètres |  Accuracy | Retrieval P@1 | Drift | Description                       |
| ----------------- | ---------: | --------: | ------------: | ----: | --------------------------------- |
| **Zero-shot**     |          0 |     0.500 |         0.400 | 0.000 | CLIP sans fine-tuning             |
| **A1 LinearHead** |      5 130 |     0.670 |         0.400 | 0.000 | Tête de classification uniquement |
| **A2 LoRA V+T**   |    983 040 | **0.874** |     **0.900** | 0.240 | LoRA sur vision + texte           |
| **B LoRA V only** |    491 520 |     0.863 |         0.800 | 0.232 | LoRA sur vision uniquement        |

### Variantes expérimentées

Pour l'étude d'ablation, plusieurs valeurs du poids de régularisation ont également été testées :

```text
A2 λ = 0.25
A2 λ = 0.50
A2 λ = 1.00
```

---

# 📊 Résultats principaux

## Gain en classification

```text
Zero-shot       ████████                    0.500
A1 LinearHead   ███████████                 0.670   +17.0 pts
B LoRA V only   ██████████████              0.863   +36.2 pts
A2 LoRA V+T     ██████████████▌             0.874   +37.4 pts ⭐
```

### Lecture

Le fine-tuning LoRA apporte un gain très important par rapport au CLIP zero-shot.

**A2 LoRA V+T** obtient la meilleure accuracy :

```text
0.500 → 0.874
```

soit :

```text
+37.4 points
```

---

## ⚖️ Trade-off paramètres ↔ performance

### A1 LinearHead

```text
5K paramètres
↓
Très faible coût d'adaptation
↓
Accuracy : 0.670
```

### B LoRA V only

```text
491K paramètres
↓
Bon compromis coût / performance
↓
Accuracy : 0.863
```

### A2 LoRA V+T

```text
983K paramètres
↓
Performance maximale
↓
Accuracy : 0.874
↓
Drift plus important
```

**Conclusion expérimentale :**

> B LoRA V only constitue un excellent compromis entre performance, coût paramétrique et préservation des représentations.

---

# 🔬 Détails techniques

## Architecture CLIP

| Élément             | Configuration                  |
| ------------------- | ------------------------------ |
| Backbone            | `openai/clip-vit-base-patch32` |
| Architecture vision | ViT-B/32                       |
| Dimension embedding | 512                            |
| Image size          | 224 × 224                      |
| Contexte texte      | 77 tokens                      |

---

## Implémentation LoRA

| Paramètre         | Configuration    |
| ----------------- | ---------------- |
| Rank              | `r=8`            |
| Variante minimale | `r=1`            |
| Modules ciblés    | Q, K, V, O       |
| MLP               | Oui              |
| Alpha             | 1.0 par défaut   |
| Ablation          | 0.25 / 0.5 / 1.0 |
| Injection         | Manuelle         |

L'implémentation LoRA est réalisée **sans bibliothèque PEFT**, afin de garder un contrôle explicite sur les modules adaptés.

---

## Entraînement

| Paramètre                  | Valeur        |
| -------------------------- | ------------- |
| Optimizer                  | AdamW         |
| Learning rate — LinearHead | `1e-4`        |
| Learning rate — LoRA       | `5e-5`        |
| Batch size                 | 64            |
| Epochs                     | 5             |
| Seeds                      | 42, 123, 2024 |

---

# 📐 Méthodologie d'évaluation

### Classification

Accuracy **Top-1** calculée sur un sous-ensemble de :

```text
2 000 images Fashion-MNIST
```

---

### Retrieval

La métrique principale est :

```text
P@1 — Precision at 1
```

pour la recherche texte → image.

---

### Drift

Le drift est mesuré à partir de la variation des représentations entre le modèle avant et après fine-tuning.

Métrique utilisée :

```text
Distance cosine moyenne
```

---

### Calibration

La calibration est évaluée avec :

```text
ECE — Expected Calibration Error
```

sur **15 bins**.

---

# 🧪 Tests automatisés

Lancer l'ensemble des tests :

```bash
pytest tests/ -v
```

Lancer un fichier spécifique :

```bash
pytest tests/test_retrieval.py -v
```

Générer un rapport HTML :

```bash
pytest tests/ --html=reports/test_report.html --self-contained-html
```

Lancer uniquement les tests rapides :

```bash
pytest tests/ -v -m "not slow"
```

### Résultat attendu

```text
==================== 53 passed in ~40s ====================
```

---

## Couverture des tests

| Fichier                  | Tests | Couverture                                       |
| ------------------------ | ----: | ------------------------------------------------ |
| `test_retrieval.py`      |    20 | Retriever, multi-modèle, métriques, accuracy@5   |
| `test_classification.py` |    11 | Classification, multi-classification, embeddings |
| `test_analytics.py`      |    12 | Team table, figures, CSV, calibration            |
| `test_consistency.py`    |     5 | Labels ↔ images ↔ embeddings                     |
| `test_app_smoke.py`      |     5 | Imports, build_app, chemins                      |

**Total : 53 tests**

---

# 🛠️ Scripts utilitaires

| Script                           | Rôle                                      |
| -------------------------------- | ----------------------------------------- |
| `scripts/prepare_eval_images.py` | Génère les 2 000 images et embeddings     |
| `scripts/regenerate_labels.py`   | Régénère `eval_labels.npy`                |
| `scripts/audit.py`               | Vérifie la cohérence des assets           |
| `scripts/run_tests.ps1`          | Lance les tests et génère le rapport HTML |

---

# 🐛 Bugs connus & limitations

| Limitation                                             | Statut    | Solution / Contournement                            |
| ------------------------------------------------------ | --------- | --------------------------------------------------- |
| Seul `clip_image_zs.npy` est généré initialement       | ✅ Attendu | Lancer `prepare_eval_images.py --model a2_lora_vt`  |
| Chargement des 4 modèles au démarrage                  | ✅ Normal  | Environ 30 s selon la machine                       |
| Inférence GPU non forcée                               | ✅ Normal  | `DEVICE = "cuda"` si CUDA est disponible            |
| Temps de démarrage plus long lors du premier lancement | ✅ Normal  | Les modèles sont chargés une seule fois par session |

---

# 🎓 Contexte académique

## Question de recherche

> **Jusqu'où peut-on adapter CLIP à un domaine spécifique comme Fashion-MNIST sans perdre sa généralité cross-modale ?**

L'étude cherche donc à mesurer simultanément :

```text
Adaptation
    ↕
Performance de classification
    ↕
Retrieval
    ↕
Préservation des représentations
    ↕
Budget de paramètres
```

---

## 🔎 Principaux enseignements

### 1. LoRA améliore fortement la classification

A2 LoRA V+T atteint :

```text
0.874 accuracy
```

contre :

```text
0.500
```

pour le modèle zero-shot.

Soit un gain de :

```text
+37.4 points
```

---

### 2. Vision-only conserve une grande partie du gain

B LoRA V only atteint :

```text
0.863 accuracy
```

avec environ **50 % des paramètres** de A2.

Il conserve ainsi environ **98 % de l'accuracy de A2** tout en utilisant deux fois moins de paramètres.

---

### 3. Le fine-tuning modifie les représentations

Le **drift angulaire** permet de quantifier objectivement la modification des embeddings après adaptation.

Cela permet d'étudier non seulement :

```text
"Le modèle est-il meilleur ?"
```

mais également :

```text
"Qu'a-t-il perdu ou modifié pour devenir meilleur ?"
```

---

### 4. Le meilleur modèle dépend du critère

Il n'existe pas nécessairement un unique modèle optimal.

```text
Performance maximale
        ↓
A2 LoRA V+T

Meilleur compromis
        ↓
B LoRA V only

Budget minimal
        ↓
A1 LinearHead
```

Cette comparaison permet donc d'étudier le fine-tuning comme un problème de **trade-off** plutôt que comme une simple optimisation de l'accuracy.

---

# 📚 Références

### CLIP

Radford, A. et al.
**Learning Transferable Visual Models From Natural Language Supervision.**
ICML, 2021.

### LoRA

Hu, E. J. et al.
**LoRA: Low-Rank Adaptation of Large Language Models.**
ICLR, 2022.

### Fashion-MNIST

Xiao, H. et al.
**Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.**
2017.

### Gradio

Abid, A. et al.
**Gradio: Hassle-Free Sharing and Testing of Machine Learning Models in the Wild.**
2019.

---

# 👤 Auteur

**Wael G.**

Développement · Fine-tuning · Expérimentation · Analyse · Interface utilisateur

Projet réalisé dans le cadre d'un **cursus académique en Data Science / Intelligence Artificielle**.

---

# 📄 Licence

Ce projet est distribué sous licence **MIT**.

Voir le fichier [`LICENSE`](LICENSE) pour les détails.

---

<p align="center">

### 🎨 CLIP LoRA Studio

**Fine-tuning · Retrieval · Cross-Modal Alignment · Representation Drift**

<br>

<b>✅ Phase 4 complète</b>

<br>

11 sections · 53 tests · 4 modèles · 2 000 images

<br><br>

<i>Projet académique — 2026</i>

</p>
