"""
modules/retrieval.py
====================
Image Retrieval — Recherche d'images similaires dans Fashion-MNIST.
Utilise la similarité cosinus sur les embeddings CLIP.

FIX (Bug 1): Support multi-modèle avec dict d'embeddings.
             Le retriever charge uniquement les fichiers disponibles.
FIX (Bug 3): Remplace les print par logging structuré.
"""
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

from utils.constants import CLASSES

# FIX (Bug 3): Logger structuré
logger = logging.getLogger(__name__)

# FIX (Bug 1): Mapping canonique modèle → fichier d'embeddings
DEFAULT_EMBEDDINGS_MAP: Dict[str, str] = {
    "Zero-shot":     "assets/data/embeddings/clip_image_zs.npy",
    "A1 LinearHead": "assets/data/embeddings/clip_image_a1.npy",
    "A2 LoRA V+T":   "assets/data/embeddings/clip_image_a2.npy",
    "B LoRA V only": "assets/data/embeddings/clip_image_b.npy",
}


class ImageRetriever:
    """
    Recherche d'images similaires dans un dataset pré-calculé.

    Accepte un dictionnaire modèle→fichier d'embeddings.
    Seuls les fichiers existants sont chargés.
    `available_models` expose la liste des modèles utilisables.

    Usage:
        retriever = ImageRetriever(
            embeddings_map={
                "Zero-shot":   "assets/data/embeddings/clip_image_zs.npy",
                "A2 LoRA V+T": "assets/data/embeddings/clip_image_a2.npy",
            },
            labels_path="assets/data/embeddings/eval_labels.npy",
            images_dir="assets/eval_images",
        )
        results = retriever.retrieve(query_embedding, model_name="Zero-shot", k=12)
    """

    def __init__(
        self,
        embeddings_map: Optional[Dict[str, str]] = None,
        labels_path: str = "assets/data/embeddings/eval_labels.npy",
        images_dir: str = "assets/eval_images",
        # Legacy compat: single embeddings_path still works
        embeddings_path: Optional[str] = None,
    ):
        # FIX (Bug 1): Build map from either dict or legacy single path
        if embeddings_map is not None:
            self._embeddings_map = {k: Path(v) for k, v in embeddings_map.items()}
        elif embeddings_path is not None:
            # Legacy: wrap into dict, detect model name from filename
            p = Path(embeddings_path)
            model_name = "Zero-shot" if "zs" in p.stem else p.stem
            self._embeddings_map = {model_name: p}
        else:
            self._embeddings_map = {k: Path(v) for k, v in DEFAULT_EMBEDDINGS_MAP.items()}

        self.labels_path = Path(labels_path)
        self.images_dir  = Path(images_dir)

        self._embeddings: Dict[str, np.ndarray] = {}   # model_name → array
        self.labels: Optional[np.ndarray] = None
        self.image_paths: List[Path] = []

        self._load_data()

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Charge embeddings (fichiers existants uniquement), labels, images."""
        logger.info("=" * 60)
        logger.info("Loading ImageRetriever")

        # 1) Embeddings — charge seulement les fichiers présents
        for model_name, path in self._embeddings_map.items():
            if path.exists():
                arr = np.load(path).astype(np.float32)
                # FIX (Bug 1): Normalise L2 au chargement pour garantir la cohérence
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                arr = arr / norms
                self._embeddings[model_name] = arr
                logger.info(
                    "Loaded embeddings [%s]: shape=%s, normalized=True",
                    model_name, arr.shape,
                )
            else:
                logger.warning(
                    "Embeddings not found for model '%s' at %s. "
                    "Run: python scripts/prepare_eval_images.py --model %s",
                    model_name, path,
                    model_name.lower().replace(" ", "_"),
                )

        if not self._embeddings:
            raise FileNotFoundError(
                "No embeddings found. "
                "Run: python scripts/prepare_eval_images.py"
            )

        # 2) Labels — recherche dans plusieurs emplacements
        possible_labels = [
            self.labels_path,
            self.images_dir / "labels.npy",
            self.images_dir.parent / "eval_labels.npy",
        ]
        for lp in possible_labels:
            if lp.exists():
                self.labels = np.load(lp).astype(np.int64)
                logger.info("Labels loaded: shape=%s from %s", self.labels.shape, lp)
                break

        if self.labels is None:
            logger.warning(
                "Labels not found — deriving from FashionMNIST. "
                "Run scripts/regenerate_labels.py for a faster start."
            )
            try:
                from torchvision.datasets import FashionMNIST
                from torchvision import transforms
                eval_indices = np.load("assets/data/eval_indices.npy")
                ds = FashionMNIST(
                    root="./data", train=False, download=True,
                    transform=transforms.ToTensor(),
                )
                self.labels = np.array([ds[int(i)][1] for i in eval_indices], dtype=np.int64)
                logger.info("Labels derived from FashionMNIST: shape=%s", self.labels.shape)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load labels. Run scripts/regenerate_labels.py. Error: {exc}"
                ) from exc

        # 3) Cohérence embeddings ↔ labels
        for model_name, arr in self._embeddings.items():
            if len(arr) != len(self.labels):
                raise ValueError(
                    f"Shape mismatch for model '{model_name}': "
                    f"{len(arr)} embeddings vs {len(self.labels)} labels"
                )

        # 4) Chemins images
        if self.images_dir.exists():
            self.image_paths = sorted(self.images_dir.glob("eval_*.png"))
            logger.info("Image paths: %d files", len(self.image_paths))
        else:
            logger.warning(
                "Images dir not found: %s. Run scripts/prepare_eval_images.py",
                self.images_dir,
            )
            self.image_paths = []

        logger.info(
            "ImageRetriever ready: %d models, %d images",
            len(self._embeddings), len(self.labels),
        )

    # ------------------------------------------------------------------
    @property
    def available_models(self) -> List[str]:
        """Liste des modèles pour lesquels des embeddings sont disponibles."""
        return list(self._embeddings.keys())

    @property
    def num_images(self) -> int:
        """Nombre d'images indexées."""
        first = next(iter(self._embeddings.values()), None)
        return len(first) if first is not None else 0

    def get_unavailable_models(self) -> Dict[str, str]:
        """Retourne les modèles non dispo avec commande de génération."""
        result = {}
        for model_name, path in self._embeddings_map.items():
            if model_name not in self._embeddings:
                slug = model_name.lower().replace(" ", "_").replace("+", "").replace("/", "")
                result[model_name] = (
                    f"Embeddings pour '{model_name}' non générés. "
                    f"Lance : python scripts/prepare_eval_images.py --model {slug}"
                )
        return result

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: np.ndarray,
        model_name: Optional[str] = None,
        k: int = 12,
        return_images: bool = True,
        class_filter: Optional[List[int]] = None,
        metric: str = "cosine",
    ) -> List[Dict]:
        """
        Trouve les k images les plus similaires à query_embedding.

        Args:
            query_embedding: np.array shape (512,) ou (1, 512).
            model_name: modèle dont les embeddings sont utilisés pour la recherche.
                        Si None, utilise le premier modèle disponible.
            k: nombre d'images à retourner.
            return_images: si True, charge les images PIL.
            class_filter: si fourni, filtre les résultats aux class_ids listés.
            metric: "cosine" | "dot" | "euclidean".

        Returns:
            Liste de dicts : {index, class, class_id, score, image?, image_path?}
        """
        # Sélection du modèle
        if model_name is None:
            model_name = self.available_models[0]

        if model_name not in self._embeddings:
            unavail = self.get_unavailable_models()
            hint = unavail.get(model_name, f"Modèle '{model_name}' inconnu.")
            raise ValueError(hint)

        db_embs = self._embeddings[model_name]  # shape (N, 512)

        # Normaliser le query
        q = np.array(query_embedding, dtype=np.float32)
        if q.ndim == 2:
            q = q[0]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Similar score
        if metric == "cosine" or metric == "dot":
            sims = db_embs @ q                           # (N,)
        elif metric == "euclidean":
            diffs = db_embs - q[np.newaxis, :]
            sims = -np.linalg.norm(diffs, axis=1)        # (N,) — négatif pour argsort
        else:
            raise ValueError(f"Métrique inconnue : {metric}. Choisir parmi cosine/dot/euclidean.")

        # Filtre par classe
        if class_filter is not None:
            mask = np.isin(self.labels, class_filter)
            sims = np.where(mask, sims, -np.inf)

        # Top-K
        top_k_idx = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_k_idx:
            idx = int(idx)
            score = float(sims[idx])
            if score == -np.inf:
                continue  # filtré

            result = {
                "index":    idx,
                "class":    CLASSES[self.labels[idx]],
                "class_id": int(self.labels[idx]),
                "score":    score,
            }

            if return_images and idx < len(self.image_paths):
                img_path = self.image_paths[idx]
                if img_path.exists():
                    result["image"] = Image.open(img_path).convert("RGB")
                    result["image_path"] = str(img_path)

            results.append(result)

        return results

    # ------------------------------------------------------------------
    def get_class_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Compte combien d'images de chaque classe dans les résultats."""
        dist: Dict[str, int] = {}
        for r in results:
            dist[r["class"]] = dist.get(r["class"], 0) + 1
        return dist

    def check_consistency(self, n_probe: int = 10, k: int = 5) -> Dict[str, float]:
        """
        Vérifie la cohérence embeddings ↔ labels par accuracy@k sur n_probe requêtes.

        Returns:
            Dict model_name → accuracy@k
        """
        np.random.seed(42)
        query_ids = np.random.choice(self.num_images, min(n_probe, self.num_images), replace=False)
        results = {}

        for model_name, db_embs in self._embeddings.items():
            hits = 0
            for qid in query_ids:
                q_label = int(self.labels[qid])
                sims = db_embs @ db_embs[qid]
                sims[qid] = -1.0   # exclure soi-même
                top_k = np.argsort(sims)[::-1][:k]
                if q_label in self.labels[top_k]:
                    hits += 1
            acc = hits / len(query_ids)
            results[model_name] = acc
            logger.info(
                "Consistency check [%s] accuracy@%d on %d probes: %.1f%%",
                model_name, k, n_probe, acc * 100,
            )

        return results