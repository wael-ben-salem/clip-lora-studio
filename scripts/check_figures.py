from pathlib import Path

FIGURES_DIR = Path("assets/figures")
figs_to_check = [
    ("TEAM_FINAL_TABLE_heatmap.png", "team"),
    ("SLIDE_CONCLUSION.png", "team"),
    ("team_robustness.png", "team"),
    ("team_catastrophic_forgetting.png", "team"),
    ("team_multi_seed.png", "team"),
    ("team_equal_params.png", "team"),
    ("team_calibration_reliability.png", "team"),
    ("team_confidence_distribution.png", "team"),
]
print("=== Figure Presence Check ===")
for name, subdir in figs_to_check:
    path = FIGURES_DIR / subdir / name
    status = "OK" if path.exists() else "MISSING"
    print(f"  [{status}] {name}")
