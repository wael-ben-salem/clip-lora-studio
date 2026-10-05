"""
scripts/test_retrieval_coherence.py
=====================================
Vérifie accuracy@K du retriever Zero-shot et rejoue le scénario
"sneaker → retrieval" pour confirmer que Bug 1 est corrigé.
"""
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
SNEAKER_ID = CLASSES.index("Sneaker")

def main():
    print("=" * 60)
    print("TEST RETRIEVAL COHERENCE")
    print("=" * 60)

    embs   = np.load(ROOT / "assets/data/embeddings/clip_image_zs.npy").astype(np.float32)
    labels = np.load(ROOT / "assets/data/embeddings/eval_labels.npy").astype(np.int64)
    images = sorted((ROOT / "assets/eval_images").glob("eval_*.png"))

    # Normalize (already done at load in new retrieval.py)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms == 0, 1, norms)

    print(f"\nEmbeddings: {embs.shape} | Labels: {labels.shape}")
    print(f"Normalized: min_norm={np.linalg.norm(embs[:5], axis=1).min():.4f}")

    # === Scenario "sneaker → retrieval" ===
    print("\n--- Scénario SNEAKER → Retrieval (Top-5) ---")
    sneaker_idxs = np.where(labels == SNEAKER_ID)[0]
    query_idx = sneaker_idxs[0]
    q = embs[query_idx]

    sims = embs @ q
    sims[query_idx] = -1.0
    top5 = np.argsort(sims)[::-1][:5]
    print(f"Query: idx={query_idx}, true_class=Sneaker")
    for rank, idx in enumerate(top5):
        cls = CLASSES[labels[idx]]
        print(f"  Top-{rank+1}: {cls} (sim={sims[idx]:.4f})")

    sneaker_in_top5 = sum(labels[i] == SNEAKER_ID for i in top5)
    print(f"\n  → {sneaker_in_top5}/5 Sneaker dans Top-5 "
          f"({'PASS' if sneaker_in_top5 >= 3 else 'WARN'})")

    # === Accuracy@K sur 50 requêtes ===
    print("\n--- Accuracy@K (50 requêtes aléatoires) ---")
    np.random.seed(42)
    query_ids = np.random.choice(len(embs), 50, replace=False)
    acc1, acc3, acc5 = [], [], []
    for qid in query_ids:
        q_label = int(labels[qid])
        sims_q = embs @ embs[qid]
        sims_q[qid] = -1.0
        top_k = np.argsort(sims_q)[::-1][:5]
        acc1.append(int(labels[top_k[0]] == q_label))
        acc3.append(int(q_label in labels[top_k[:3]]))
        acc5.append(int(q_label in labels[top_k[:5]]))

    A1 = np.mean(acc1) * 100
    A3 = np.mean(acc3) * 100
    A5 = np.mean(acc5) * 100
    print(f"  Accuracy@1 : {A1:.1f}%")
    print(f"  Accuracy@3 : {A3:.1f}%")
    print(f"  Accuracy@5 : {A5:.1f}%  {'[PASS]' if A5 >= 80 else '[WARN]'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
