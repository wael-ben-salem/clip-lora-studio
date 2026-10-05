# 🔧 Guide d'installation — CLIP LoRA Studio

Guide pas à pas pour installer et lancer le projet, sur **Windows**, **Linux** et **macOS**.

---

## 📋 Prérequis

| Composant | Version minimale | Recommandé |
|-----------|------------------|------------|
| Python | 3.10 | 3.11 |
| pip | 23.0 | dernière |
| RAM | 8 GB | 16 GB |
| Disque | 2 GB | 5 GB |
| CUDA (optionnel) | 11.8 | 12.1 |

**Temps d'installation estimé** : 10-15 minutes.

---

## 🚀 Installation rapide (5 minutes)

### Windows (PowerShell)

```powershell
# 1. Clone ou téléchargement du projet
git clone <ton-repo> clip-lora-studio
cd clip-lora-studio

# 2. Créer l'environnement virtuel
python -m venv Clip-lora-studio-env
.\Clip-lora-studio-env\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Préparer les données
python scripts/prepare_eval_images.py
python scripts/regenerate_labels.py

# 5. Lancer l'app
python app.py