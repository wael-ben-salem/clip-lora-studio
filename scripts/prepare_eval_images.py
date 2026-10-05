"""
Extrait les 2000 images eval de FashionMNIST et les sauvegarde en PNG.
À lancer UNE SEULE FOIS avant d'utiliser l'Image Retrieval.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm


# Chemins
ROOT = Path(__file__).parent.parent
EVAL_INDICES_PATH = ROOT / "assets" / "data" / "eval_indices.npy"
OUTPUT_DIR = ROOT / "assets" / "eval_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("📸 PRÉPARATION DES IMAGES EVAL")
    print("=" * 70)
    
    # 1) Charger les indices
    print("\n1) Chargement des indices...")
    eval_indices = np.load(EVAL_INDICES_PATH)
    print(f"   ✅ {len(eval_indices)} indices chargés")
    
    # 2) Charger FashionMNIST test (sans transformation)
    print("\n2) Chargement FashionMNIST...")
    ds = FashionMNIST(
        root=str(ROOT / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),  # [0, 1], shape (1, 28, 28)
    )
    print(f"   ✅ {len(ds)} images disponibles")
    
    # 3) Extraire les 2000 images
    print(f"\n3) Extraction vers {OUTPUT_DIR}...")
    for i, idx in enumerate(tqdm(eval_indices, desc="Extraction")):
        img_tensor, label = ds[idx]
        
        # Convertir en PIL RGB 224×224
        img_pil = transforms.ToPILImage()(img_tensor).convert("RGB")
        img_pil = img_pil.resize((224, 224), Image.LANCZOS)
        
        # Sauvegarder : eval_0000.png, eval_0001.png, ...
        out_path = OUTPUT_DIR / f"eval_{i:04d}.png"
        img_pil.save(out_path)
    
    print(f"   ✅ {len(eval_indices)} images sauvegardées")
    
    # 4) Sauvegarder les labels associés
    labels = np.array([ds[i][1] for i in eval_indices])
    labels_path = OUTPUT_DIR / "labels.npy"
    np.save(labels_path, labels)
    print(f"   ✅ Labels sauvegardés : {labels_path}")
    
    print("\n" + "=" * 70)
    print(f"🎉 Terminé ! {len(eval_indices)} images dans {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()