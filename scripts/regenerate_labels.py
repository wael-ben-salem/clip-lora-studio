"""
scripts/regenerate_labels.py
============================
Régénère eval_labels.npy depuis les indices officiels eval_indices.npy
et le dataset FashionMNIST (source de vérité).

Sortie :
  - assets/data/embeddings/eval_labels.npy  (shape: 2000,)
  - assets/eval_images/labels.npy            (copie pour accès rapide)
  - assets/debug/labels_check.png            (image de contrôle visuel)

Usage :
  python scripts/regenerate_labels.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image

# === Chemins ===
ROOT = Path(__file__).parent.parent
EVAL_INDICES_PATH = ROOT / "assets" / "data" / "eval_indices.npy"
EVAL_IMAGES_DIR   = ROOT / "assets" / "eval_images"
EMBEDDINGS_DIR    = ROOT / "assets" / "data" / "embeddings"
DEBUG_DIR         = ROOT / "assets" / "debug"
DATA_DIR_FMNIST   = ROOT / "data"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main():
    print("=" * 60)
    print("📋 RÉGÉNÉRATION eval_labels.npy")
    print("=" * 60)

    # 1) Charger les indices
    if not EVAL_INDICES_PATH.exists():
        raise FileNotFoundError(f"eval_indices.npy introuvable : {EVAL_INDICES_PATH}")
    eval_indices = np.load(EVAL_INDICES_PATH)
    print(f"\n[1] eval_indices chargés : {len(eval_indices)} indices")
    print(f"    Plage : [{eval_indices.min()} – {eval_indices.max()}]")

    # 2) Charger FashionMNIST test (source de vérité)
    print("\n[2] Chargement FashionMNIST test...")
    ds = FashionMNIST(
        root=str(DATA_DIR_FMNIST),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    print(f"    ✅ {len(ds)} images disponibles")

    # 3) Extraire les labels dans l'ordre des indices
    print("\n[3] Extraction des labels...")
    labels = np.array([ds[int(i)][1] for i in eval_indices], dtype=np.int64)
    print(f"    ✅ {len(labels)} labels extraits — shape : {labels.shape}")

    # 4) Distribution
    counts = np.bincount(labels, minlength=10)
    print("\n[4] Distribution des labels :")
    total_ok = True
    for cls_id, (cls_name, count) in enumerate(zip(CLASSES, counts)):
        status = "✅" if 150 <= count <= 250 else "⚠️"
        if count < 150 or count > 250:
            total_ok = False
        print(f"    {status} [{cls_id}] {cls_name:<14} : {count:4d} images")
    print(f"\n    Total : {labels.sum()} ≠ 0 (vrai), np.bincount sum = {counts.sum()}")
    if total_ok:
        print("    [PASS] Distribution équilibrée (~200 par classe)")
    else:
        print("    [WARN] Distribution déséquilibrée (attendu ~200 par classe)")

    # 5) Sauvegarder les labels
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    canonical_path = EMBEDDINGS_DIR / "eval_labels.npy"
    np.save(canonical_path, labels)
    print(f"\n[5] Sauvegardé → {canonical_path}")

    # Copie dans eval_images/ pour compatibilité avec le retriever
    copy_path = EVAL_IMAGES_DIR / "labels.npy"
    np.save(copy_path, labels)
    print(f"    Copié  → {copy_path}")

    # 6) Générer l'image de contrôle visuel
    print("\n[6] Génération image de contrôle labels_check.png...")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Prendre 20 images espacées uniformément
    n_check = 20
    indices_20 = np.linspace(0, len(labels) - 1, n_check, dtype=int)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(
        f"Contrôle labels_check.png — 20 images + labels\n"
        f"(rouge = label prédit par nom de fichier, vert = label FashionMNIST)",
        fontsize=12, y=1.01
    )

    for ax, img_idx in zip(axes.flat, indices_20):
        img_path = EVAL_IMAGES_DIR / f"eval_{img_idx:04d}.png"
        label_id  = int(labels[img_idx])
        label_str = CLASSES[label_id]
        fmnist_idx = int(eval_indices[img_idx])

        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            # Afficher directement depuis FashionMNIST si le PNG n'existe pas
            img_tensor = ds[fmnist_idx][0]
            ax.imshow(img_tensor.squeeze(), cmap="gray")

        ax.set_title(
            f"#{img_idx} → {label_str}\n(FashionMNIST idx: {fmnist_idx})",
            fontsize=8, color="green"
        )
        ax.axis("off")

    plt.tight_layout()
    out_path = DEBUG_DIR / "labels_check.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Sauvegardé → {out_path}")

    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print(f"   eval_labels.npy → {canonical_path}")
    print(f"   labels_check.png → {out_path}")
    print("=" * 60)
    return labels

if __name__ == "__main__":
    main()
