"""
Chargement et gestion de CLIP + LoRA.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

from utils.constants import CLIP_MODEL, DEVICE, CHECKPOINTS_DIR


class ModelRegistry:
    """
    Charge CLIP base + LoRA adapters et expose une API unifiée.
    
    Usage:
        registry = ModelRegistry().load_all()
        registry.get("A2 LoRA V+T")     # → modèle
        registry.names                  # ["Zero-shot", "A2 LoRA V+T", "B LoRA V only"]
        registry.proc                   # CLIPProcessor
        registry.device                 # "cuda" ou "cpu"
    """
    
    def __init__(self, checkpoint_dir=None):
        self.device = DEVICE
        self.checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self.models = {}
        self.clip = None
        self.proc = None
        self._loaded = False
    
    # -------------------------------------------------------------------------
    # CHARGEMENT
    # -------------------------------------------------------------------------
    def load_all(self):
        """Charge CLIP base + LoRA (injection manuelle des poids)."""
        if self._loaded:
            return self
        
        print("=" * 70)
        print("🔄 CHARGEMENT DES MODÈLES")
        print("=" * 70)
        print(f"🖥️  Device : {self.device}")
        
        # 1) CLIP base
        print(f"\n🔄 CLIP base : {CLIP_MODEL}")
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.models["Zero-shot"] = self.clip
        print("✅ CLIP base chargé")
        
        # 2) A2 LoRA V+T
        a2_path = self.checkpoint_dir / "A2_lora"
        if a2_path.exists():
            print(f"\n🔄 A2 LoRA V+T : {a2_path}")
            self.a2 = self._load_lora_manually(a2_path, "A2 LoRA V+T")
            self.models["A2 LoRA V+T"] = self.a2
            print("✅ A2 LoRA V+T chargé (injection manuelle)")
        else:
            print(f"⚠️  A2 LoRA introuvable : {a2_path}")
        
        # 3) B LoRA V only
        b_path = self.checkpoint_dir / "B_lora"
        if b_path.exists():
            print(f"\n🔄 B LoRA V only : {b_path}")
            self.b_model = self._load_lora_manually(b_path, "B LoRA V only")
            self.models["B LoRA V only"] = self.b_model
            print("✅ B LoRA V only chargé (injection manuelle)")
        else:
            print(f"⚠️  B LoRA introuvable : {b_path}")
        
        self._loaded = True
        print("\n" + "=" * 70)
        print(f"✅ {len(self.models)} modèle(s) chargé(s) : {list(self.models.keys())}")
        print("=" * 70)
        return self
    def _load_lora_manually(self, lora_path, name):
        """Charge un LoRA en INJECTANT DIRECTEMENT les poids dans CLIP.
        
        Méthode : on calcule delta_W = lora_B @ lora_A * (alpha / r)
        et on l'ajoute directement aux poids des couches Linear.
        """
        import json
        from safetensors.torch import load_file as safe_load
        
        # 1) Charger la config
        with open(lora_path / "adapter_config.json") as f:
            cfg = json.load(f)
        
        r           = cfg["r"]
        alpha       = cfg["lora_alpha"]
        scaling     = alpha / r
        target_mods = cfg["target_modules"]
        
        # 2) Charger les poids LoRA
        state_dict = safe_load(str(lora_path / "adapter_model.safetensors"))
        
        # 3) Charger un CLIP frais
        base = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        
        # 4) 🔑 Regrouper les paires lora_A / lora_B par couche cible
        # Exemple de clé : "base_model.model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"
        lora_pairs = {}
        for k, v in state_dict.items():
            # Extraire le chemin de la couche cible
            # Enlever le suffixe ".lora_X.weight"
            if ".lora_A.weight" in k:
                layer_path = k.replace(".lora_A.weight", "")
                if layer_path not in lora_pairs:
                    lora_pairs[layer_path] = {}
                lora_pairs[layer_path]["A"] = v
            elif ".lora_B.weight" in k:
                layer_path = k.replace(".lora_B.weight", "")
                if layer_path not in lora_pairs:
                    lora_pairs[layer_path] = {}
                lora_pairs[layer_path]["B"] = v
        
        print(f"   📊 Paires LoRA trouvées : {len(lora_pairs)}")
        
        # 5) 🔑 Injecter chaque paire dans la couche CLIP correspondante
        n_injected = 0
        for layer_path, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                print(f"   ⚠️  Paire incomplète : {layer_path}")
                continue
            
            # Retirer le préfixe "base_model.model."
            clean_path = layer_path.replace("base_model.model.", "", 1)
            # → "vision_model.encoder.layers.0.self_attn.q_proj"
            
            # Naviguer dans la hiérarchie du modèle
            parts = clean_path.split(".")
            target = base
            try:
                for part in parts:
                    target = getattr(target, part)
            except AttributeError:
                print(f"   ⚠️  Couche introuvable : {clean_path}")
                continue
            
            # Vérifier que c'est bien une Linear
            if not isinstance(target, torch.nn.Linear):
                print(f"   ⚠️  Pas une Linear : {clean_path} ({type(target).__name__})")
                continue
            
            # 🎯 Calculer delta_W = B @ A * scaling
            A = pair["A"].to(self.device)  # shape (r, in_features)
            B = pair["B"].to(self.device)  # shape (out_features, r)
            delta_W = (B @ A) * scaling    # shape (out_features, in_features)
            
            # Injecter directement
            with torch.no_grad():
                target.weight.data += delta_W
            n_injected += 1
        
        print(f"   📊 Couches modifiées   : {n_injected} / {len(lora_pairs)}")
        
        return base
    # -------------------------------------------------------------------------
    # ACCESSEURS
    # -------------------------------------------------------------------------
    def get(self, name):
        """Récupère un modèle par son nom."""
        if name not in self.models:
            raise ValueError(f"Modèle inconnu : {name}. "
                             f"Disponibles : {list(self.models.keys())}")
        return self.models[name]
    
    @property
    def names(self):
        """Liste des noms de modèles."""
        return list(self.models.keys())
    
    @property
    def is_loaded(self):
        return self._loaded