import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    else:
        print(f"[FAIL] {message}")
        return False

def main():
    print("="*50)
    print(" PART 1 - COMPREHENSIVE AUDIT REPORT")
    print("="*50)
    
    project_root = Path(".")
    
    # 1.1 File inventory
    print("\n--- 1.1 FILE INVENTORY ---")
    for root, dirs, files in os.walk(project_root):
        # Exclude common ignore dirs for cleaner output
        if any(ignored in root for ignored in [".git", "__pycache__", ".venv", "venv", ".pytest_cache"]):
            continue
        rel_path = Path(root).relative_to(project_root)
        if rel_path == Path("."):
            print(f"./")
        else:
             print(f"{rel_path}/")
        
        for file in files:
            print(f"  └── {file}")
            
    # 1.2 Verification of Assets & 1.3 Consistency Checks
    print("\n--- 1.2 & 1.3 ASSET VERIFICATION & CONSISTENCY ---")
    
    assets_dir = project_root / "assets"
    
    # Check expected embeddings and labels
    embeddings_path = assets_dir / "data" / "embeddings" / "clip_image_zs.npy"
    labels_path = assets_dir / "data" / "embeddings" / "eval_labels.npy"
    
    embeddings = None
    labels = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"Shape of clip_image_zs.npy: {embeddings.shape}")
        
        # Check L2 norm
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"L2 Norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
        is_normalized = np.allclose(norms, 1.0, atol=1e-3)
        check(is_normalized, "Embeddings are L2 normalized")
    else:
        check(False, f"Found {embeddings_path}")
        
    if labels_path.exists():
        labels = np.load(labels_path)
        print(f"Shape of eval_labels.npy: {labels.shape}")
        
        dist = Counter(labels)
        print(f"Distribution of labels: {dict(dist.most_common())}")
    else:
        check(False, f"Found {labels_path}")
        
    if embeddings is not None and labels is not None:
         check(embeddings.shape[0] == labels.shape[0], "Consistency: Embeddings shape matches Labels shape")
         
    # Check eval images
    eval_images_dir = assets_dir / "eval_images"
    eval_images = []
    if eval_images_dir.exists():
        eval_images = list(eval_images_dir.glob("*.png")) + list(eval_images_dir.glob("*.jpg"))
        print(f"Number of images in assets/eval_images/: {len(eval_images)}")
        
        if labels is not None:
            check(len(eval_images) == len(labels), "Consistency: Number of eval_images matches labels count")
    else:
        check(False, "Found assets/eval_images/ directory")
        
    # List Figures
    figures_team_dir = assets_dir / "figures" / "team"
    if figures_team_dir.exists():
        figures = list(figures_team_dir.glob("*.png"))
        print(f"\nFigures in assets/figures/team/ ({len(figures)}):")
        for f in figures:
             print(f"  - {f.name}")
    else:
        print("\nDirectory assets/figures/team/ not found.")
        
    # List CSVs
    data_dir = assets_dir / "data"
    if data_dir.exists():
        csvs = list(data_dir.glob("*.csv"))
        print(f"\nCSV files in assets/data/ ({len(csvs)}):")
        for c in csvs:
             print(f"  - {c.name}")
             
        # Check specific expected columns
        expected_csvs = {
            "team_final_table.csv": ["Model", "Accuracy", "Parameters"],
            "team_confusion_delta.csv": ["Class", "Delta"]  # Guessing based on common schema
        }
        for csv_name, expected_cols in expected_csvs.items():
            csv_path = data_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                found = all(col in df.columns for col in expected_cols)
                check(found, f"Expected columns in {csv_name}")
            else:
                 check(False, f"Found {csv_name}")
    else:
        print("\nDirectory assets/data/ not found.")
        
    # List example images
    examples_dir = assets_dir / "examples"
    if examples_dir.exists():
         examples = list(examples_dir.glob("*.*"))
         print(f"\nImages in assets/examples/ ({len(examples)}):")
         for e in examples:
             print(f"  - {e.name}")
             
    # Basic Main Files check
    expected_files = [
        "app.py",
        "modules/models.py",
        "modules/inference.py",
        "modules/analytics.py",
        "modules/retrieval.py",
        "utils/constants.py",
        "scripts/prepare_eval_images.py"
    ]
    
    print("\n--- 1.3 MAIN FILES PRESENCE ---")
    all_main = True
    for f in expected_files:
        if not (project_root / f).exists():
             check(False, f"Found {f}")
             all_main = False
    
    if all_main:
        check(True, "All main expected files are present")

if __name__ == "__main__":
    main()
