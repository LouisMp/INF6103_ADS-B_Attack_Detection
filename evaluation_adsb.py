#!/usr/bin/env python3
"""
Vérité terrain et métriques communes aux deux modèles (non supervisé et supervisé).

POURQUOI UNE VÉRITÉ TERRAIN « EFFECTIVE » ?
La colonne attackType du simulateur indique qu'un avion est DÉSIGNÉ pour une attaque, pas que l'attaque a
produit un effet sur le message. Exemples observés dans nos captures :
  - Jumping : l'avion ne saute qu'après son premier appel au contrôleur (arrivée dans l'espace aérien),
    puis le saut est tiré au hasard toutes les 2 s simulées. Entre deux sauts, les messages sont normaux.
  - Standing still : le gel commence à un instant aléatoire (jusqu'à 500 s simulées après l'éligibilité).
  - Un changement de réglage en cours de capture retire et redistribue les étiquettes, alors qu'un avion
    figé reste figé.
On compare donc les modèles à ce qui s'est réellement passé, défini PHYSIQUEMENT :

  verite = "Jumping aircraft"         message étiqueté Jumping ET déplacement > 3 x le déplacement attendu
                                      d'après la vitesse annoncée, ou saut d'altitude > 500 ft
  verite = "Aircraft standing still"  message étiqueté Still, à partir du 1er message immobile de l'avion
                                      (vitesse annoncée > 50 kt mais aucun déplacement)
  verite = "Trajectory modification"  tout message étiqueté VTM (le cap diffusé est modifié dès le début)
  verite = "No attack"                messages des avions JAMAIS étiquetés pendant toute la capture
  verite = "zone grise"               le reste (étiqueté mais sans effet, ou avion attaqué à un autre moment) :
                                      exclu des métriques par message

Seuls les messages évalués (avion en vol, hors pause de simulation) comptent.
Le retard de détection est en MESSAGES (le CSV n'a pas de colonne de temps).

ATTENTION À LA CIRCULARITÉ : cette vérité terrain utilise des critères physiques simples (x3, 500 ft,
immobile). Le modèle supervisé est entraîné sur elle et peut donc la « réapprendre ». La comparaison la plus
indépendante est le tableau par avion contre les étiquettes brutes du simulateur, et le taux de fausses
alertes sur les avions jamais attaqués.
"""
import numpy as np
import pandas as pd

NORMAL = "No attack"
JUMP, STILL, VTM = "Jumping aircraft", "Aircraft standing still", "Trajectory modification"
ATTAQUES = [JUMP, STILL, VTM]
CLASSES = [NORMAL] + ATTAQUES
GRIS = "zone grise"
NOMS_COURTS = {NORMAL: "Normal", JUMP: "Jumping", STILL: "Standing still", VTM: "VTM"}

SEUIL_SAUT = 3.0        # déplacement > 3 x attendu
SEUIL_SAUT_ALT = 500    # ft entre deux messages (normalisé par le time warp)
SEUIL_IMMOBILE = 0.1    # déplacement < 10 % de l'attendu


def verite_terrain(f):
    """Ajoute les colonnes `verite` et `debut_effet` (n° de message du début de l'effet, par avion)."""
    f = f.copy()
    jamais = f.groupby("id").attackType.transform(lambda s: (s == NORMAL).all())
    v = pd.Series(GRIS, index=f.index, dtype=object)
    v[jamais] = NORMAL
    v[f.attackType == VTM] = VTM

    saut = (f.attackType == JUMP) & f.evalue & ((f.pas_norm > SEUIL_SAUT) | (f.d_alt > SEUIL_SAUT_ALT))
    v[saut] = JUMP

    immobile = (f.attackType == STILL) & f.evalue & (f.pas_norm < SEUIL_IMMOBILE)
    debut_gel = f[immobile].groupby("id").k.min()
    apres_gel = f.k >= f.id.map(debut_gel)            # NaN -> False
    v[(f.attackType == STILL) & apres_gel] = STILL

    f["verite"] = v
    debut = pd.Series(np.nan, index=f.index)
    for c in ATTAQUES:
        d = f[(f.verite == c) & f.evalue].groupby("id").k.min()
        m = f.id.isin(d.index)
        debut[m] = debut[m].fillna(f.id[m].map(d))
    f["debut_effet"] = debut
    return f


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else np.nan
    return p, r, f1


def tableau_messages(f, col="prediction"):
    """Métriques par MESSAGE (zone grise exclue). `col` contient la classe prédite (NORMAL si pas d'alerte)."""
    e = f[f.evalue & (f.verite != GRIS)]
    lignes = []
    for c in ATTAQUES:
        vrai, pred = e.verite == c, e[col] == c
        tp, fp, fn = int((vrai & pred).sum()), int((~vrai & pred).sum()), int((vrai & ~pred).sum())
        p, r, f1 = _prf(tp, fp, fn)
        alerte_quelconque = int((vrai & (e[col] != NORMAL)).sum())
        lignes.append({"classe": NOMS_COURTS[c], "messages_attaqués": int(vrai.sum()), "VP": tp, "FP": fp, "FN": fn,
                       "précision": p, "rappel": r, "F1": f1,
                       "rappel_toute_alerte": alerte_quelconque / vrai.sum() if vrai.sum() else np.nan})
    vrai, pred = e.verite != NORMAL, e[col] != NORMAL
    tp, fp, fn = int((vrai & pred).sum()), int((~vrai & pred).sum()), int((vrai & ~pred).sum())
    p, r, f1 = _prf(tp, fp, fn)
    lignes.append({"classe": "Toute attaque (binaire)", "messages_attaqués": int(vrai.sum()), "VP": tp, "FP": fp,
                   "FN": fn, "précision": p, "rappel": r, "F1": f1, "rappel_toute_alerte": r})
    tab = pd.DataFrame(lignes)
    normaux = e[e.verite == NORMAL]
    fausses = int((normaux[col] != NORMAL).sum())
    info = {"messages_normaux": len(normaux), "fausses_alertes": fausses,
            "fausses_alertes_pour_1000": 1000 * fausses / max(len(normaux), 1)}
    return tab, info


def tableau_avions(f, col="prediction"):
    """Métriques par AVION.
    - avec effet : avions dont l'attaque a réellement produit un effet (vérité terrain) ; détecté = au moins
      une alerte (tout type) sur un message attaqué ; retard = messages entre le début de l'effet et la 1re alerte.
    - étiquetés (brut) : avions étiquetés en vol par le simulateur ; détecté = au moins une alerte pendant
      que l'étiquette est présente. Inclut les avions désignés mais jamais touchés par l'effet.
    - normal : avions jamais étiquetés ; « détecté » = au moins une fausse alerte."""
    e = f[f.evalue]
    alerte = e[col] != NORMAL
    lignes = []
    for c in ATTAQUES:
        att = e[e.verite == c]
        ids_eff = att.id.unique()
        debuts = att.groupby("id").k.min()                 # début de l'effet DE CETTE attaque
        det_eff, retards, bon_type = 0, [], 0
        for i in ids_eff:
            a = e[e.id == i]
            debut = debuts[i]
            al = a[(a[col] != NORMAL) & (a.verite == c)]
            if len(al):
                det_eff += 1
                bon_type += int((al[col] == c).any())
                retards.append(int(al.k.min() - debut))
        lab = e[e.attackType == c]
        ids_lab = lab.id.unique()
        det_lab = int(lab[alerte.loc[lab.index]].id.nunique())
        lignes.append({
            "attaque": NOMS_COURTS[c],
            "avions_étiquetés_en_vol": len(ids_lab), "détectés_parmi_étiquetés": det_lab,
            "taux_étiquetés": det_lab / len(ids_lab) if len(ids_lab) else np.nan,
            "avions_avec_effet": len(ids_eff), "détectés_parmi_effet": det_eff,
            "taux_effet": det_eff / len(ids_eff) if len(ids_eff) else np.nan,
            "bon_type": bon_type,
            "retard_médian_msgs": float(np.median(retards)) if retards else np.nan,
            "retard_max_msgs": float(np.max(retards)) if retards else np.nan,
        })
    norm = e[e.verite == NORMAL]
    ids_n = norm.id.unique()
    fp_ids = int(norm[norm[col] != NORMAL].id.nunique())
    lignes.append({"attaque": "Normal (jamais attaqué)", "avions_étiquetés_en_vol": len(ids_n),
                   "détectés_parmi_étiquetés": fp_ids, "taux_étiquetés": fp_ids / len(ids_n) if len(ids_n) else np.nan,
                   "avions_avec_effet": np.nan, "détectés_parmi_effet": np.nan, "taux_effet": np.nan,
                   "bon_type": np.nan, "retard_médian_msgs": np.nan, "retard_max_msgs": np.nan})
    return pd.DataFrame(lignes)


def matrice_confusion(f, col="prediction"):
    e = f[f.evalue & (f.verite != GRIS)]
    m = pd.crosstab(e.verite.map(NOMS_COURTS), e[col].map(NOMS_COURTS), rownames=["vérité"], colnames=["prédit"])
    ordre = [NOMS_COURTS[c] for c in CLASSES]
    return m.reindex(index=ordre, columns=ordre, fill_value=0)


def evaluer(f, col="prediction", afficher=True):
    """Ajoute la vérité terrain si besoin et retourne (tableau messages, info, tableau avions, confusion)."""
    if "verite" not in f:
        f = verite_terrain(f)
    tm, info = tableau_messages(f, col)
    ta = tableau_avions(f, col)
    cm = matrice_confusion(f, col)
    if afficher:
        pd.set_option("display.width", 200)
        print("Par message (zone grise exclue) :")
        print(tm.round(3).to_string(index=False))
        print(f"Fausses alertes : {info['fausses_alertes']} sur {info['messages_normaux']} messages normaux "
              f"({info['fausses_alertes_pour_1000']:.2f} pour 1000)\n")
        print("Par avion :")
        print(ta.round(3).to_string(index=False))
        print("\nMatrice de confusion (messages) :")
        print(cm.to_string())
    return tm, info, ta, cm
