#!/usr/bin/env python3
"""
Reproduit tous les résultats : entraîne les deux modèles et produit les tableaux d'évaluation.

    python experiences.py

Protocole
  1. Validation croisée à 5 plis GROUPÉE PAR AVION sur capture_A_normal + capture_B1_entrainement :
     un avion est entièrement dans le pli d'entraînement ou entièrement dans le pli de test (jamais de fuite
     de messages d'un même vol). Les plis sont stratifiés sur la classe principale de l'avion.
  2. Test indépendant : modèles finaux entraînés sur A + B1, évalués sur capture_B0_test (autre partie,
     réglages modifiés en cours de route, donc étiquettes plus bruitées).
Le non supervisé n'apprend que sur les avions normaux ; le supervisé apprend sur toutes les classes.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import detecteur_adsb as det
import evaluation_adsb as ev
import supervise_adsb as sup

ICI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ICI, "data")
RES = os.path.join(ICI, "resultats")
MOD = os.path.join(ICI, "modeles")
ENTRAINEMENT = [os.path.join(DATA, "capture_A_normal.csv.gz"), os.path.join(DATA, "capture_B1_entrainement.csv.gz")]
TEST = os.path.join(DATA, "capture_B0_test.csv.gz")
MODELES = {"non_supervise": "Non supervisé (enveloppe normale)", "supervise": "Supervisé (forêt aléatoire)"}


def classe_principale(f):
    """Classe d'un avion pour stratifier les plis : son attaque effective, sinon normal / zone grise."""
    def c(s):
        att = [x for x in ev.ATTAQUES if (s == x).any()]
        return att[0] if att else (ev.NORMAL if (s == ev.NORMAL).all() else ev.GRIS)
    return f.groupby("id").verite.agg(c)


def validation_croisee(F, k=5):
    cls = classe_principale(F)
    ids = cls.index.values
    plis = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    F = F.copy()
    F["pred_non_supervise"] = ev.NORMAL
    F["pred_supervise"] = ev.NORMAL
    for n, (itr, ite) in enumerate(plis.split(ids, cls.values, groups=ids), 1):
        tr, te = F.id.isin(ids[itr]), F.id.isin(ids[ite])
        m_ns = det.entrainer(F[tr], verbeux=False)
        F.loc[te, "pred_non_supervise"] = det.detecter(F[te], m_ns).prediction.values
        m_s = sup.entrainer(F[tr], verbeux=False)
        F.loc[te, "pred_supervise"] = sup.detecter(F[te], m_s).prediction.values
        print(f"  pli {n}/{k} : {len(ids[ite])} avions en test")
    return F


def tableau_comparatif(f):
    """Tableau compact pour les diapositives : une ligne par attaque, les deux modèles côte à côte."""
    lignes = []
    for cle, nom in MODELES.items():
        tm, info, ta, _ = ev.evaluer(f, f"pred_{cle}", afficher=False)
        tm, ta = tm.set_index("classe"), ta.set_index("attaque")
        for c in ev.ATTAQUES:
            n = ev.NOMS_COURTS[c]
            lignes.append({"modèle": nom, "attaque": n,
                           "précision": tm.loc[n, "précision"], "rappel": tm.loc[n, "rappel"], "F1": tm.loc[n, "F1"],
                           "avions détectés (effet)": f"{int(ta.loc[n, 'détectés_parmi_effet'])}/{int(ta.loc[n, 'avions_avec_effet'])}",
                           "avions détectés (étiquette brute)": f"{int(ta.loc[n, 'détectés_parmi_étiquetés'])}/{int(ta.loc[n, 'avions_étiquetés_en_vol'])}",
                           "retard médian (msgs)": ta.loc[n, "retard_médian_msgs"]})
        b = tm.loc["Toute attaque (binaire)"]
        nrm = ta.loc["Normal (jamais attaqué)"]
        lignes.append({"modèle": nom, "attaque": "Toute attaque (binaire)",
                       "précision": b["précision"], "rappel": b["rappel"], "F1": b["F1"],
                       "avions détectés (effet)": "",
                       "avions détectés (étiquette brute)": "",
                       "retard médian (msgs)": np.nan,
                       "fausses alertes / 1000 msgs normaux": info["fausses_alertes_pour_1000"],
                       "avions normaux avec fausse alerte": f"{int(nrm['détectés_parmi_étiquetés'])}/{int(nrm['avions_étiquetés_en_vol'])}"})
    return pd.DataFrame(lignes)


def _md(df, flottants=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else (f"{x:.{flottants}f}" if abs(x) < 10 else f"{x:.1f}"))
    d = d.fillna("")
    lignes = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lignes += ["| " + " | ".join(map(str, r)) + " |" for r in d.values]
    return "\n".join(lignes)


def ecrire(f, nom_protocole, titre, md):
    md.append(f"\n## {titre}\n")
    comp = tableau_comparatif(f)
    comp.to_csv(os.path.join(RES, f"comparatif_{nom_protocole}.csv"), index=False)
    md.append("### Tableau comparatif\n")
    md.append(_md(comp))
    for cle, nom in MODELES.items():
        tm, info, ta, cm = ev.evaluer(f, f"pred_{cle}", afficher=False)
        tm.to_csv(os.path.join(RES, f"messages_{nom_protocole}_{cle}.csv"), index=False)
        ta.to_csv(os.path.join(RES, f"avions_{nom_protocole}_{cle}.csv"), index=False)
        cm.to_csv(os.path.join(RES, f"confusion_{nom_protocole}_{cle}.csv"))
        md.append(f"\n### {nom}\n")
        md.append("Par message (zone grise exclue) :\n")
        md.append(_md(tm))
        md.append(f"\nFausses alertes : **{info['fausses_alertes']}** sur {info['messages_normaux']} messages normaux "
                  f"(**{info['fausses_alertes_pour_1000']:.2f} pour 1000**).\n")
        md.append("Par avion :\n")
        md.append(_md(ta))
        md.append("\nMatrice de confusion (messages ; lignes = vérité, colonnes = prédiction) :\n")
        md.append(_md(cm.reset_index()))
    return comp


def figure(comp_cv, comp_test, sortie):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, TXT, TXT2, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    couleurs = {MODELES["non_supervise"]: "#2a78d6", MODELES["supervise"]: "#eb6834"}
    classes = ["Jumping", "Standing still", "VTM", "Toute attaque (binaire)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURF, sharey=True)
    for ax, comp, titre in [(axes[0], comp_cv, "Validation croisée (A + B1, 5 plis par avion)"),
                            (axes[1], comp_test, "Test indépendant (capture B0)")]:
        x = np.arange(len(classes))
        for j, (nom, coul) in enumerate(couleurs.items()):
            v = comp[comp["modèle"] == nom].set_index("attaque").loc[classes, "F1"].values
            b = ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=coul, label=nom, edgecolor=SURF, linewidth=2)
            for xi, vi in zip(x + (j - 0.5) * 0.36, v):
                ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=8.5, color=TXT)
        ax.set_xticks(x, ["Jumping", "Standing\nstill", "VTM", "Toute\nattaque"])
        ax.set_ylim(0, 1.12)
        ax.set_facecolor(SURF)
        ax.set_title(titre, loc="left", fontsize=10.5, color=TXT)
        ax.grid(True, axis="y", color=GR, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GR)
        ax.tick_params(colors=TXT2, labelsize=9)
    axes[0].set_ylabel("F1 par message", color=TXT2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9.5, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Détection des attaques ADS-B : F1 par type d'attaque", x=0.01, ha="left", fontsize=13, color=TXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(sortie, dpi=150, facecolor=SURF)


def figures_exemples(T, modele, dossier):
    """Une figure par attaque : l'avion de la capture de test dont l'effet dure le plus longtemps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = det.detecter(T, modele)
    res["verite"] = T.verite.values
    for c, nom in [(ev.JUMP, "jumping"), (ev.STILL, "standing_still"), (ev.VTM, "vtm")]:
        n = res[(res.verite == c) & res.evalue].groupby("id").size()
        if n.empty:
            continue
        ident = n.idxmax()
        fig = det.tracer_avion(res, ident, modele, sortie=os.path.join(dossier, f"exemple_{nom}.png"))
        plt.close(fig)
    fig = det.tracer_vue_ensemble(res, os.path.join(dossier, "vue_ensemble_test.png"))
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(MOD, exist_ok=True)
    print("Lecture des captures ...")
    F = ev.verite_terrain(det.charger(ENTRAINEMENT))
    T = ev.verite_terrain(det.charger(TEST))
    T["capture"] = "capture_B0_test"

    print("Validation croisée ...")
    Fcv = validation_croisee(F)

    print("Modèles finaux (entraînés sur A + B1) ...")
    m_ns = det.entrainer(F, os.path.join(MOD, "modele_normal.json"))
    det.afficher_modele(m_ns)
    m_s = sup.entrainer(F, os.path.join(MOD, "modele_supervise.joblib"))
    T["pred_non_supervise"] = det.detecter(T, m_ns).prediction.values
    T["pred_supervise"] = sup.detecter(T, m_s).prediction.values

    md = ["# Résultats d'évaluation\n",
          "Généré par `python experiences.py`. Définition de la vérité terrain : voir `evaluation_adsb.py`.\n",
          "## Données\n"]
    donnees = []
    for nom, f in [("A + B1 (entraînement / validation croisée)", F), ("B0 (test)", T)]:
        e = f[f.evalue]
        donnees.append({"jeu": nom, "messages": len(f), "avions": f.id.nunique(),
                        "messages en vol évalués": len(e),
                        **{f"vérité {ev.NOMS_COURTS.get(c, c)}": int((e.verite == c).sum()) for c in ev.CLASSES + [ev.GRIS]}})
    md.append(_md(pd.DataFrame(donnees)))
    comp_cv = ecrire(Fcv, "cv", "1. Validation croisée groupée par avion (A + B1)", md)
    comp_test = ecrire(T, "test", "2. Test indépendant (B0)", md)
    figure(comp_cv, comp_test, os.path.join(RES, "figures", "f1_par_attaque.png"))
    figures_exemples(T, m_ns, os.path.join(RES, "figures"))
    imp = pd.Series(m_s["foret"].feature_importances_, index=sup.VARIABLES).sort_values(ascending=False)
    md.append("\n## Importance des variables (forêt aléatoire finale)\n")
    md.append(_md(imp.rename("importance").reset_index().rename(columns={"index": "variable"})))
    md.append("\n## Seuils appris (non supervisé final)\n")
    md.append("```json\n" + json.dumps({"seuils": m_ns["seuils"], "apprentissage": m_ns["apprentissage"]},
                                        indent=2, ensure_ascii=False) + "\n```\n")
    with open(os.path.join(RES, "RESULTATS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\nRésultats écrits dans resultats/RESULTATS.md")
    print("\n=== Validation croisée ===\n", comp_cv.round(3).to_string(index=False))
    print("\n=== Test B0 ===\n", comp_test.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
