"""
Fonctions d'inférence CLIP + LoRA.
"""

import torch
import numpy as np
import pandas as pd

from utils.constants import CLASSES, CLASS_PROMPTS


# =============================================================================
# ENCODAGE
# =============================================================================
@torch.no_grad()
def get_image_embedding(registry, model_name, image):
    """
    Encode une image en embedding L2-normalisé.
    
    Args:
        registry : ModelRegistry chargé
        model_name : str
        image : PIL.Image
    
    Returns:
        torch.Tensor de shape (1, 512), L2-normalisé
    """
    model = registry.get(model_name)
    image = image.convert("RGB").resize((224, 224))
    inputs = registry.proc(images=image, return_tensors="pt").to(registry.device)
    emb = model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def get_text_embeddings(registry, model_name, prompts):
    """
    Encode une liste de textes en embeddings L2-normalisés.
    
    Returns:
        torch.Tensor de shape (N, 512), L2-normalisé
    """
    model = registry.get(model_name)
    inputs = registry.proc(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(registry.device)
    emb = model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)


# =============================================================================
# CLASSIFICATION
# =============================================================================
@torch.no_grad()
def classify(registry, image, model_name="A2 LoRA V+T", top_k=5,
             custom_prompts=None, temperature=100.0):
    """
    Classifie une image avec le modèle spécifié.
    
    Args:
        registry : ModelRegistry
        image : PIL.Image
        model_name : str
        top_k : int
        custom_prompts : list[str] (optionnel) — prompts personnalisés
        temperature : float — softmax temperature (10.0 par défaut)
    
    Returns:
        (top_dict, df)
        - top_dict : {"T-shirt/top": 0.85, "Shirt": 0.10, ...}
        - df : DataFrame avec Class, Probability, Similarity
    """
    if image is None:
        return None, pd.DataFrame()
    
    prompts = custom_prompts if custom_prompts else CLASS_PROMPTS
    labels = CLASSES if not custom_prompts else [f"Prompt {i+1}" for i in range(len(prompts))]
    
    # Encodage
    img_emb = get_image_embedding(registry, model_name, image)
    txt_emb = get_text_embeddings(registry, model_name, prompts)
    
    # Similarités cosinus
    sims = (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()
    
    # Softmax avec température
    exp_sims = np.exp(sims * temperature)
    probs = exp_sims / exp_sims.sum()
    
    # Top-K
    top_idx = probs.argsort()[::-1][:top_k]
    top_dict = {labels[i]: float(probs[i]) for i in top_idx}
    
    # DataFrame complet
    df = pd.DataFrame({
        "Class": labels,
        "Probability": probs,
        "Similarity": sims,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)
    
    return top_dict, df


# =============================================================================
# CLASSIFICATION MULTI-MODÈLES (pour comparateur)
# =============================================================================
@torch.no_grad()
def classify_multi(registry, image, model_names=None, top_k=1):
    """
    Classifie une image avec plusieurs modèles et retourne un comparatif.
    
    Returns:
        pd.DataFrame avec colonnes: Model, Top Class, Confidence
    """
    if image is None:
        return pd.DataFrame()
    
    if model_names is None:
        model_names = registry.names
    
    rows = []
    for name in model_names:
        top, _ = classify(registry, image, model_name=name, top_k=top_k)
        if top:
            top_class = list(top.keys())[0]
            confidence = list(top.values())[0]
            rows.append({
                "Model": name,
                "Prediction": top_class,
                "Confidence": f"{confidence:.4f}",
            })
    
    return pd.DataFrame(rows)