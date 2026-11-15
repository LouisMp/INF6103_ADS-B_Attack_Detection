#!/usr/bin/env python3
"""
Détecteur ADS-B SUPERVISÉ (forêt aléatoire) pour OpenScope-sec.

IDÉE : mêmes variables physiques que le détecteur non supervisé (detecteur_adsb.py), mais le modèle apprend à
partir d'exemples étiquetés de chaque attaque. Il prédit, pour chaque message d'un avion en vol, une classe parmi
    No attack | Jumping aircraft | Aircraft standing still | Trajectory modification

Étiquettes d'entraînement = vérité terrain « effective » (voir evaluation_adsb.py) : la colonne attackType du
simulateur, nettoyée des messages où l'attaque n'a produit aucun effet (zone grise, exclue de l'apprentissage).

Commandes :
    python supervise_adsb.py entrainer  capture1.csv [capture2.csv ...]  --modele modeles/modele_supervise.joblib
    python supervise_adsb.py detecter   capture.csv  --modele modeles/modele_supervise.joblib

Dépendances : pandas, numpy, scikit-learn, joblib.
Le fichier .joblib est un pickle : ne chargez que des modèles produits par vous-même.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import detecteur_adsb as det
import evaluation_adsb as ev

NORMAL = ev.NORMAL
VARIABLES = ["pas_norm", "fige_run", "d_cap", "d_alt", "cap_abs"]
PARAMS_FORET = {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 5,
                "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": 0}


def _x(f):
    X = f[VARIABLES].astype(float).copy()
    X["pas_norm"] = X.pas_norm.clip(upper=50)        # un saut de 30x ou 300x, c'est la même information
    X["d_alt"] = X.d_alt.clip(upper=5000)
    return X


def exemples(f):
    """Messages utilisables pour l'apprentissage : en vol, hors zone grise."""
    if "verite" not in f:
        f = ev.verite_terrain(f)
    return f[f.evalue & (f.verite != ev.GRIS)]


def entrainer(source, chemin_modele=None, params=None, verbeux=True):
    """`source` : chemin(s) de CSV étiquetés, ou DataFrame déjà passé par detecteur_adsb.charger."""
    f = source if isinstance(source, pd.DataFrame) and "pas_norm" in source else det.charger(source)
    e = exemples(f)
    foret = RandomForestClassifier(**{**PARAMS_FORET, **(params or {})})
    foret.fit(_x(e), e.verite)
    modele = {"type": "foret_aleatoire", "version": 1, "variables": VARIABLES, "foret": foret,
              "params_variables": det.PARAMS_DEFAUT,
              "apprentissage": {"captures": sorted(f.capture.unique()) if "capture" in f else [],
                                "messages": int(len(e)),
                                "par_classe": e.verite.value_counts().to_dict(),
                                "avions": int(e.id.nunique())}}
    if verbeux:
        print(f"Appris sur {len(e)} messages / {e.id.nunique()} avions : {e.verite.value_counts().to_dict()}")
        imp = pd.Series(foret.feature_importances_, index=VARIABLES).sort_values(ascending=False)
        print("Importance des variables :", imp.round(3).to_dict())
    if chemin_modele:
        joblib.dump(modele, chemin_modele, compress=3)
        if verbeux:
            print(f"Modèle enregistré : {chemin_modele}")
    return modele


def charger_modele(chemin):
    return joblib.load(chemin)


def detecter(source, modele):
    """Retourne un DataFrame (un message par ligne) avec `prediction`, `proba_attaque` et `alerte`."""
    if isinstance(modele, str):
        modele = charger_modele(modele)
    f = source.copy() if isinstance(source, pd.DataFrame) and "pas_norm" in source \
        else det.charger(source, modele.get("params_variables"))
    f["prediction"] = NORMAL
    f["proba_attaque"] = 0.0
    m = f.evalue
    if m.any():
        foret = modele["foret"]
        X = _x(f[m])
        f.loc[m, "prediction"] = foret.predict(X)
        proba = foret.predict_proba(X)
        idx_normal = list(foret.classes_).index(NORMAL)
        f.loc[m, "proba_attaque"] = 1 - proba[:, idx_normal]
    f["alerte"] = f.prediction != NORMAL
    precedent = f.groupby("id").alerte.shift(1) == True          # noqa: E712
    f["debut_alerte"] = f.alerte & ~precedent
    return f


def main():
    ap = argparse.ArgumentParser(description="Détecteur ADS-B supervisé (OpenScope-sec)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("entrainer")
    e.add_argument("csv", nargs="+")
    e.add_argument("--modele", default="modeles/modele_supervise.joblib")
    d = sp.add_parser("detecter")
    d.add_argument("csv")
    d.add_argument("--modele", default="modeles/modele_supervise.joblib")
    args = ap.parse_args()
    if args.cmd == "entrainer":
        entrainer(args.csv, args.modele)
        return
    res = detecter(args.csv, args.modele)
    al = res[res.alerte]
    print(f"{res.id.nunique()} avions, {len(res)} messages, {len(al)} messages en alerte, {al.id.nunique()} avions concernés")
    print(al.groupby("id").agg(callsign=("callsign", "first"), alertes=("alerte", "size"),
                               type=("prediction", lambda s: s.mode().iloc[0]))
            .sort_values("alertes", ascending=False).head(15).to_string())
    if (res.attackType != NORMAL).any():
        print("\n=== Comparaison avec les étiquettes du simulateur ===")
        ev.evaluer(res, "prediction")


if __name__ == "__main__":
    main()
