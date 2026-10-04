"""
Génère 10 images Fashion-MNIST d'exemple pour la démo Gradio.
"""
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from PIL import Image

# Dossier de sortie
OUT = Path("assets/examples")
OUT.mkdir(parents=True, exist_ok=True)

# Charger Fashion-MNIST test
ds = FashionMNIST(root="./data", train=False, download=True,
                  transform=transforms.ToTensor())

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Prendre 1 image par classe
print("📸 Génération des exemples...")
picked = {}

for img, label in ds:
    if label not in picked:
        picked[label] = img
        print(f"   ✓ {CLASSES[label]:15s} trouvé")
    if len(picked) == 10:
        break

# Sauvegarder en PNG haute résolution
for label, tensor in picked.items():
    # tensor : (1, 28, 28) → (28, 28) → RGB upscalé
    img_pil = transforms.ToPILImage()(tensor).convert("RGB")
    img_pil = img_pil.resize((224, 224), Image.LANCZOS)
    
    name = CLASSES[label].replace("/", "_").lower()
    path = OUT / f"{name}.png"
    img_pil.save(path)
    print(f"💾 {path}")

print(f"\n✅ {len(picked)} images générées dans {OUT}")