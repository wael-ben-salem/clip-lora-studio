"""
🔬 DIAGNOSTIC V2 — Utilise la classe ModelRegistry (avec le fix)
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from modules.models import ModelRegistry
from modules.inference import classify


print("=" * 70)
print("🔬 DIAGNOSTIC V2 — Via ModelRegistry")
print("=" * 70)

# =============================================================================
# 1) Charger via ta classe corrigée
# =============================================================================
print("\n[1] Chargement via ModelRegistry...")
registry = ModelRegistry().load_all()
print(f"   ✅ Modèles : {registry.names}")

# =============================================================================
# 2) Vérifier que A2 et B sont bien des objets PeftModel (pas CLIP)
# =============================================================================
print("\n[2] Type des modèles :")
for name in registry.names:
    m = registry.get(name)
    print(f"   {name:20s} → {type(m).__name__}")

# =============================================================================
# 3) Vérifier les paramètres LoRA dans A2 et B
# =============================================================================
print("\n[3] Paramètres LoRA :")
for name in ["A2 LoRA V+T", "B LoRA V only"]:
    m = registry.get(name)
    total = 0
    nonzero = 0
    for n, p in m.named_parameters():
        if "lora_" in n:
            total += p.numel()
            nonzero += (p != 0).sum().item()
    print(f"   {name:20s} : {total:,} params ({nonzero:,} non-zéro)")

# =============================================================================
# 4) Test sur une image
# =============================================================================
print("\n[4] Test sur une image :")
img_path = list(Path("assets/examples").glob("*.png"))[0]
print(f"   📸 {img_path.name}")
image = Image.open(img_path)

for model_name in registry.names:
    top, df = classify(registry, image, model_name=model_name, top_k=3)
    top_class = list(top.keys())[0]
    conf = list(top.values())[0]
    print(f"   {model_name:20s} → {top_class:15s} ({conf*100:.1f}%)")

# =============================================================================
# 5) Vérification critique : les embeddings diffèrent ?
# =============================================================================
print("\n[5] Vérification embeddings :")
from modules.inference import get_image_embedding

embs = {}
for model_name in registry.names:
    embs[model_name] = get_image_embedding(registry, model_name, image).cpu()

print(f"   ZS vs A2 : cos = {(embs['Zero-shot'] * embs['A2 LoRA V+T']).sum().item():.6f}")
print(f"   ZS vs B  : cos = {(embs['Zero-shot'] * embs['B LoRA V only']).sum().item():.6f}")

if (embs['Zero-shot'] * embs['A2 LoRA V+T']).sum().item() < 0.95:
    print("   ✅ Les embeddings DIFFÈRENT → LoRA est appliqué !")
else:
    print("   ❌ Les embeddings sont IDENTIQUES → LoRA n'est PAS appliqué")

print("\n" + "=" * 70)