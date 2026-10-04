"""Inspecte le contenu du fichier adapter_model.safetensors."""
from safetensors import safe_open
from pathlib import Path

for name in ["A2_lora", "B_lora"]:
    path = Path(f"checkpoints/{name}/adapter_model.safetensors")
    print(f"\n{'='*70}")
    print(f"📦 {name}")
    print(f"{'='*70}")
    print(f"   Taille : {path.stat().st_size / 1e6:.2f} MB")
    
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        print(f"   Nombre de clés : {len(keys)}")
        print(f"\n   5 premières clés :")
        for k in keys[:5]:
            tensor = f.get_tensor(k)
            print(f"      • {k}")
            print(f"        shape={tuple(tensor.shape)}, "
                  f"mean={tensor.float().mean().item():.6f}, "
                  f"std={tensor.float().std().item():.6f}")
        
        print(f"\n   5 dernières clés :")
        for k in keys[-5:]:
            print(f"      • {k}")