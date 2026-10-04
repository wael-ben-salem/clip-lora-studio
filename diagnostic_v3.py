"""🔬 Diagnostic ULTIME — Voir les clés attendues par PEFT."""
import json
from pathlib import Path
from safetensors.torch import load_file as safe_load
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


CLIP_MODEL = "openai/clip-vit-base-patch32"

print("=" * 70)
print("🔬 DIAGNOSTIC V3 — Comparaison des clés")
print("=" * 70)

# =============================================================================
# 1) Charger les clés du safetensors
# =============================================================================
print("\n[1] Clés du fichier safetensors :")
state_dict = safe_load("checkpoints/A2_lora/adapter_model.safetensors")
keys_file = list(state_dict.keys())
print(f"   Total : {len(keys_file)} clés")
print(f"   Exemples :")
for k in keys_file[:3]:
    print(f"      • {k}")

# =============================================================================
# 2) Créer un PEFT model vide et voir les clés attendues
# =============================================================================
print("\n[2] Création d'un PEFT model vide :")
with open("checkpoints/A2_lora/adapter_config.json") as f:
    cfg = json.load(f)

base = CLIPModel.from_pretrained(CLIP_MODEL)
lora_config = LoraConfig(
    r=cfg["r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg.get("lora_dropout", 0.0),
    bias=cfg.get("bias", "none"),
    target_modules=cfg["target_modules"],
)
peft_model = get_peft_model(base, lora_config)

# Récupérer les clés attendues par PEFT
keys_expected = [n for n, p in peft_model.named_parameters() if "lora_" in n]
print(f"   Total : {len(keys_expected)} clés LoRA attendues")
print(f"   Exemples :")
for k in keys_expected[:3]:
    print(f"      • {k}")

# =============================================================================
# 3) Comparer les 2 ensembles
# =============================================================================
print("\n[3] Comparaison :")
set_file     = set(keys_file)
set_expected = set(keys_expected)

only_in_file     = set_file - set_expected
only_in_expected = set_expected - set_file
common           = set_file & set_expected

print(f"   Clés en commun        : {len(common)}")
print(f"   Clés uniquement dans safetensors : {len(only_in_file)}")
print(f"   Clés uniquement attendues        : {len(only_in_expected)}")

# Exemples de clés uniquement dans safetensors
if only_in_file:
    print(f"\n   Exemples de clés UNIQUEMENT dans safetensors (5 premières) :")
    for k in list(only_in_file)[:5]:
        print(f"      • {k}")

if only_in_expected:
    print(f"\n   Exemples de clés UNIQUEMENT attendues (5 premières) :")
    for k in list(only_in_expected)[:5]:
        print(f"      • {k}")

# =============================================================================
# 4) Tester différents préfixes
# =============================================================================
print("\n[4] Test de transformation des clés :")

# Option A : tel quel
keys_A = keys_file

# Option B : retirer "base_model.model."
keys_B = [k.replace("base_model.model.", "", 1) for k in keys_file]

# Option C : ajouter "base_model.model." si absent
keys_C = [k if k.startswith("base_model.model.") else f"base_model.model.{k}" for k in keys_file]

for label, keys in [("A: tel quel", keys_A), ("B: -base_model.model.", keys_B), ("C: +base_model.model.", keys_C)]:
    common_count = len(set(keys) & set_expected)
    print(f"   {label:30s} → {common_count} clés en commun / {len(keys_expected)}")

# =============================================================================
# 5) Voir les 5 dernières clés attendues
# =============================================================================
print("\n[5] 5 dernières clés ATTENDUES par PEFT :")
for k in keys_expected[-5:]:
    print(f"      • {k}")

print("\n[6] 5 dernières clés du FICHIER :")
for k in keys_file[-5:]:
    print(f"      • {k}")

print("\n" + "=" * 70)