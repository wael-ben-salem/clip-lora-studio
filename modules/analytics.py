"""
Chargement des données d'analyse (JSON/CSV du notebook).
"""

import json
import pandas as pd
from pathlib import Path

from utils.constants import DATA_DIR, FIGURES_DIR


class Analytics:
    """
    Charge et cache les résultats d'analyse du notebook.
    
    Usage:
        a = Analytics()
        a.team_table          # DataFrame
        a.calibration         # dict
        a.robustness          # DataFrame
        a.catastrophic        # dict
    """
    
    def __init__(self, data_dir=None, figures_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self._cache = {}
    
    # -------------------------------------------------------------------------
    # CHARGEMENT JSON
    # -------------------------------------------------------------------------
    def _load_json(self, name, subdir=""):
        key = f"json:{subdir}/{name}"
        if key in self._cache:
            return self._cache[key]
        
        path = self.data_dir / subdir / f"{name}.json" if subdir else self.data_dir / f"{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = None
        
        self._cache[key] = data
        return data
    
    # -------------------------------------------------------------------------
    # CHARGEMENT CSV
    # -------------------------------------------------------------------------
    def _load_csv(self, name, subdir=""):
        key = f"csv:{subdir}/{name}"
        if key in self._cache:
            return self._cache[key]
        
        path = self.data_dir / subdir / f"{name}.csv" if subdir else self.data_dir / f"{name}.csv"
        if path.exists():
            data = pd.read_csv(path)
        else:
            data = pd.DataFrame()
        
        self._cache[key] = data
        return data
    
    # -------------------------------------------------------------------------
    # CHARGEMENT TXT
    # -------------------------------------------------------------------------
    def _load_txt(self, name, subdir=""):
        key = f"txt:{subdir}/{name}"
        if key in self._cache:
            return self._cache[key]
        
        path = self.data_dir / subdir / f"{name}.txt" if subdir else self.data_dir / f"{name}.txt"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = f.read()
        else:
            data = ""
        
        self._cache[key] = data
        return data
    
    # -------------------------------------------------------------------------
    # PROPRIÉTÉS — DONNÉES
    # -------------------------------------------------------------------------
    @property
    def team_table(self):
        """DataFrame TEAM_FINAL_TABLE.csv"""
        return self._load_csv("TEAM_FINAL_TABLE")
    
    @property
    def calibration(self):
        """Dict team_calibration.json"""
        return self._load_json("team_calibration")
    
    @property
    def confusion_delta(self):
        """DataFrame team_confusion_delta.csv"""
        return self._load_csv("team_confusion_delta")
    
    @property
    def robustness(self):
        """DataFrame team_robustness.csv"""
        return self._load_csv("team_robustness")
    
    @property
    def catastrophic(self):
        """Dict team_catastrophic_forgetting.json"""
        return self._load_json("team_catastrophic_forgetting")
    
    @property
    def multi_seed(self):
        """DataFrame team_multi_seed.csv"""
        return self._load_csv("team_multi_seed")
    
    @property
    def equal_params(self):
        """DataFrame team_equal_params.csv"""
        return self._load_csv("team_equal_params")
    
    @property
    def limitations(self):
        """Texte LIMITATIONS.txt"""
        return self._load_txt("LIMITATIONS")
    
    @property
    def final_summary(self):
        """Texte FINAL_SUMMARY.txt"""
        return self._load_txt("FINAL_SUMMARY")
    
    @property
    def conclusion(self):
        """Texte TEAM_conclusion.txt"""
        return self._load_txt("TEAM_conclusion", subdir="team")
    
    # -------------------------------------------------------------------------
    # MÉTHODES UTILITAIRES
    # -------------------------------------------------------------------------
    def get_figure(self, name, subdir="team"):
        """Retourne le chemin d'une figure si elle existe."""
        path = self.figures_dir / subdir / name
        return str(path) if path.exists() else None
    
    def list_available_figures(self, subdir="team"):
        """Liste les figures disponibles."""
        folder = self.figures_dir / subdir
        if not folder.exists():
            return []
        return sorted([f.name for f in folder.glob("*.png")])
    
    def list_available_data(self):
        """Liste les fichiers de données disponibles."""
        if not self.data_dir.exists():
            return []
        return sorted([f.name for f in self.data_dir.glob("*")
                       if f.suffix in [".json", ".csv", ".txt"]])