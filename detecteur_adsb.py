#!/usr/bin/env python3
"""
Détecteur d'anomalies ADS-B (non supervisé) pour OpenScope-sec.

IDÉE : on apprend ce qu'est un vol NORMAL (enveloppe de variables physiques), puis on signale tout
message qui sort de cette enveloppe. Aucune attaque n'est utilisée pour l'entraînement.

Deux commandes (un ou plusieurs CSV, .csv ou .csv.gz) :
    python detecteur_adsb.py entrainer  normal1.csv [normal2.csv ...]  --modele modeles/modele_normal.json
    python detecteur_adsb.py detecter   capture.csv  --modele modeles/modele_normal.json [--id aircraft-123]

Dépendances : pandas, numpy, matplotlib   (pip install -r requirements.txt)

Variables calculées pour chaque message d'un avion en vol (vitesse sol > VITESSE_MIN_KT) :
    pas_norm  distance parcourue depuis le message précédent, divisée par la distance attendue
              d'après la vitesse sol annoncée. ≈ 1 en vol normal, quelle que soit la vitesse de
              simulation (time warp).      grand -> saut de position (Jumping)
                                           ≈ 0   -> avion figé (Standing still)
    fige_run  nombre de messages consécutifs sans déplacement alors que la vitesse annoncée est > 0
    d_cap     variation, sur FENETRE_CAP messages, de l'écart (cap annoncé - trajectoire réelle),
              relativisée par le changement de trajectoire (un virage normal ne compte pas)
                                           grand -> cap falsifié (Trajectory modification)
    d_alt     variation d'altitude entre deux messages (ft, normalisée par le time warp)
                                           grand -> altitude brusque (typique d'un saut)
    cap_abs   écart absolu (degrés) entre le cap annoncé et la trajectoire sol. En vol normal il reste
              petit (angle de dérive dû au vent).   grand -> cap falsifié, y compris le plateau du VTM

ENTRAÎNEMENT : seuls les avions jamais étiquetés comme attaqués servent, et parmi eux on écarte les avions
« contaminés » (physiquement impossibles : figés >= 3 messages alors que la vitesse annoncée est > 0, ou
déplacement > 3 fois le déplacement attendu). Ils sont listés à l'écran et dans le modèle.

LIMITES CONNUES
  - Il faut le CSV « Download ads-b » (tous les avions) : les pauses de simulation et le time warp sont
    estimés avec la population d'avions. Avec le CSV d'un seul avion, une pause ressemble à un gel et les
    variations de cadence de l'affichage peuvent donner de fausses alertes « saut de position ».
  - cap_abs dépend du vent de l'aéroport : un VTM plus petit que la dérive normale (ex. 1°) n'est vu que
    pendant sa rampe (d_cap).
  - Les avions au sol (vitesse sol <= VITESSE_MIN_KT) ne sont pas évalués.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

NORMAL = "No attack"            # étiquette des messages normaux dans la colonne attackType
KT_EN_MS = 0.514444             # 1 nœud en m/s
RAYON_TERRE_M = 6_371_000

# --- Paramètres d'apprentissage (modifiables, enregistrés dans le modèle) --------------------------------
PARAMS_DEFAUT = {
    "vitesse_min_kt": 50,       # en dessous : avion au sol, non évalué
    "pas_msg_s": 40 / 60,       # durée simulée entre 2 messages à 1x (1 message / 40 mises à jour à 60 ips)
    "fenetre_cap": 8,           # messages comparés pour d_cap
    "marge": 1.5,               # seuil = marge x valeur extrême vue sur le trafic normal ...
    "planchers": {              # ... mais jamais en dessous de ces valeurs (sens physique)
        "pas_norm": 0.2,        # écart toléré autour de 1 (= 20 % de la distance attendue)
        "fige_run": 3,          # messages consécutifs immobiles
        "d_cap": 0.5,
        "d_alt": 100,
        "cap_abs": 2.0,         # degrés
    },
}
VARIABLES = ["pas_norm", "fige_run", "d_cap", "d_alt", "cap_abs"]


# =====================================================================================================
# 1. Lecture et variables
# =====================================================================================================
def _sources(source):
    """Normalise l'entrée : un chemin, un DataFrame, ou une liste des deux."""
    return list(source) if isinstance(source, (list, tuple)) else [source]


def _nom(source, n):
    if isinstance(source, pd.DataFrame):
        return f"capture{n}"
    nom = str(source).replace("\\", "/").split("/")[-1]
    for ext in (".gz", ".csv"):
        nom = nom[: -len(ext)] if nom.endswith(ext) else nom
    return nom


def lire(source):
    """Accepte un chemin de CSV (.csv ou .csv.gz) ou un DataFrame."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    manquantes = {"id", "latitude", "longitude", "altitude", "heading", "groundTrack", "groundSpeed"} - set(df.columns)
    if manquantes:
        sys.exit(f"Colonnes manquantes dans le CSV : {sorted(manquantes)}")
    if "attackType" not in df.columns:
        df["attackType"] = NORMAL
    df["attackType"] = df["attackType"].fillna(NORMAL)
    if "callsign" not in df.columns:
        df["callsign"] = df["id"]
    return df.reset_index(drop=True)


def _distance_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * RAYON_TERRE_M * np.arcsin(np.sqrt(a))


def _numeroter_instants(ids):
    """Le simulateur écrit, à chaque instant, un message par avion. Un nouvel instant commence quand
    un identifiant déjà vu dans le bloc courant revient."""
    vus, instant, sortie = set(), 0, np.empty(len(ids), dtype=int)
    for n, i in enumerate(ids):
        if i in vus:
            instant += 1
            vus = set()
        vus.add(i)
        sortie[n] = instant
    return sortie


def calculer_variables(df, params=None):
    p = {**PARAMS_DEFAUT, **(params or {})}
    f = df.copy().reset_index(drop=True)
    f["instant"] = _numeroter_instants(f["id"].values)
    g = f.groupby("id", sort=False)
    f["k"] = g.cumcount()                                    # n° du message pour cet avion

    f["pas_m"] = _distance_m(g.latitude.shift(), g.longitude.shift(), f.latitude, f.longitude)
    en_vol = f.groundSpeed > p["vitesse_min_kt"]
    attendu_m = f.groundSpeed * KT_EN_MS * p["pas_msg_s"]    # distance attendue à 1x
    ratio = (f.pas_m / attendu_m).where(en_vol)              # ≈ facteur de time warp (1, 2, 5...)

    # Facteur d'échelle de temps de chaque instant = médiane de tous les avions en vol à cet instant.
    # S'il y a moins de 5 avions en vol, on se rabat sur la médiane récente de l'avion lui-même.
    population = ratio.groupby(f.instant).transform("median")
    nb_en_vol = en_vol.groupby(f.instant).transform("sum")
    propre = ratio.groupby(f.id).transform(lambda s: s.rolling(15, min_periods=3).median().shift())
    echelle = population.where(nb_en_vol >= 5, propre)
    pause = (nb_en_vol >= 5) & (population < 0.2)            # tous les avions immobiles = simulation en pause
    echelle = echelle.where(echelle > 0.2)
    echelle = echelle.groupby(f.id).ffill()                  # garde l'échelle si elle s'effondre (avion figé)

    f["pause"] = pause
    f["echelle"] = echelle
    f["evalue"] = en_vol & echelle.notna() & f.pas_m.notna() & ~pause

    f["pas_norm"] = ratio / echelle
    fige = en_vol & ~pause & (f.pas_norm < 0.1)
    serie = fige.groupby([f.id, (~fige).groupby(f.id).cumsum()]).cumsum()
    f["fige_run"] = serie.clip(upper=30)

    ecart = np.degrees((f.heading - f.groundTrack + np.pi) % (2 * np.pi) - np.pi)
    f["ecart_cap"] = ecart
    var_ecart = (ecart - ecart.groupby(f.id).shift(p["fenetre_cap"])).abs()
    var_traj = np.degrees((f.groundTrack - f.groupby("id").groundTrack.shift(p["fenetre_cap"]) + np.pi)
                          % (2 * np.pi) - np.pi).abs()
    f["d_cap"] = (var_ecart / (var_traj + 1.0)).clip(upper=10)

    f["d_alt"] = f.groupby("id", sort=False).altitude.diff().abs() / echelle
    f["cap_abs"] = ecart.abs()

    f["pas_norm"] = f.pas_norm.fillna(1.0)
    for c in ["fige_run", "d_cap", "d_alt", "cap_abs"]:
        f[c] = f[c].fillna(0.0)
    return f


def charger(source, params=None):
    """Lit un ou plusieurs CSV et calcule les variables FICHIER PAR FICHIER (les instants et le time warp
    sont propres à chaque capture). Avec plusieurs fichiers, les identifiants sont préfixés par le nom du
    fichier pour éviter les collisions (aircraft-123 peut exister dans deux captures)."""
    srcs = _sources(source)
    morceaux = []
    for n, s in enumerate(srcs):
        f = calculer_variables(lire(s), params)
        f["capture"] = _nom(s, n)
        if len(srcs) > 1:
            f["id"] = f["capture"] + ":" + f["id"].astype(str)
        morceaux.append(f)
    return pd.concat(morceaux, ignore_index=True) if len(morceaux) > 1 else morceaux[0]


def avions_normaux_propres(f):
    """Avions jamais étiquetés comme attaqués, sans comportement physiquement impossible.
    Retourne (ids propres, ids contaminés écartés)."""
    jamais = f.groupby("id").attackType.apply(lambda s: (s == NORMAL).all())
    candidats = f[f.id.isin(jamais[jamais].index) & f.evalue]
    impossible = (candidats.fige_run >= 3) | (candidats.pas_norm > 3)
    contamines = sorted(candidats[impossible].id.unique())
    propres = sorted(set(jamais[jamais].index) - set(contamines))
    return propres, contamines


# =====================================================================================================
# 2. Entraînement : apprendre l'enveloppe du trafic normal
# =====================================================================================================
def entrainer(source, chemin_modele=None, params=None, variables=None, verbeux=True):
    """Apprend les seuils sur les avions JAMAIS attaqués et non contaminés.
    `source` : un chemin, un DataFrame, une liste de chemins, ou un DataFrame déjà passé par `charger`."""
    p = {**PARAMS_DEFAUT, **(params or {})}
    p["planchers"] = {**PARAMS_DEFAUT["planchers"], **(params or {}).get("planchers", {})}
    f = source if isinstance(source, pd.DataFrame) and "pas_norm" in source else charger(source, p)
    propres, contamines = avions_normaux_propres(f)
    if contamines and verbeux:
        print(f"{len(contamines)} avion(s) 'normal' écarté(s) de l'apprentissage (comportement impossible) : "
              f"{', '.join(contamines[:10])}{' ...' if len(contamines) > 10 else ''}")
    X = f[f.id.isin(propres) & f.evalue]
    if len(X) < 500:
        print(f"ATTENTION : seulement {len(X)} messages normaux en vol pour apprendre ; capturez un run plus long.")

    mediane = float(X.pas_norm.median())
    pl, m = p["planchers"], p["marge"]
    seuils = {
        "pas_norm_bas": mediane - max(m * (mediane - float(X.pas_norm.min())), pl["pas_norm"]),
        "pas_norm_haut": mediane + max(m * (float(X.pas_norm.max()) - mediane), pl["pas_norm"]),
        "fige_run": max(pl["fige_run"], m * float(X.fige_run.max())),
        "d_cap": max(m * float(X.d_cap.max()), pl["d_cap"]),
        "d_alt": max(m * float(X.d_alt.max()), pl["d_alt"]),
        "cap_abs": max(m * float(X.cap_abs.max()), pl["cap_abs"]),
    }
    modele = {
        "version": 2,
        "params": p,
        "seuils": {k: round(v, 4) for k, v in seuils.items()},
        "apprentissage": {
            "captures": sorted(f.capture.unique()) if "capture" in f else [],
            "avions": int(X.id.nunique()),
            "messages": int(len(X)),
            "avions_ecartes": contamines,
            "valeurs_extremes_normales": {
                "pas_norm": [round(float(X.pas_norm.min()), 4), round(float(X.pas_norm.max()), 4)],
                "fige_run": float(X.fige_run.max()),
                "d_cap": round(float(X.d_cap.max()), 4),
                "d_alt": round(float(X.d_alt.max()), 2),
                "cap_abs": round(float(X.cap_abs.max()), 3),
            },
        },
    }
    if chemin_modele:
        with open(chemin_modele, "w", encoding="utf-8") as fh:
            json.dump(modele, fh, indent=2, ensure_ascii=False)
        if verbeux:
            print(f"Modèle enregistré : {chemin_modele}")
    return modele


def charger_modele(chemin):
    with open(chemin, encoding="utf-8") as fh:
        return json.load(fh)


def afficher_modele(modele):
    s, a = modele["seuils"], modele["apprentissage"]
    print(f"Appris sur {a['avions']} avions / {a['messages']} messages normaux en vol.")
    print(f"  pas_norm   normal dans [{s['pas_norm_bas']:.2f} ; {s['pas_norm_haut']:.2f}]   (vu en normal : {a['valeurs_extremes_normales']['pas_norm']})")
    print(f"  fige_run   alerte à partir de {s['fige_run']:.0f} messages immobiles   (vu en normal : {a['valeurs_extremes_normales']['fige_run']:.0f})")
    print(f"  d_cap      alerte au-dessus de {s['d_cap']:.2f}   (vu en normal : max {a['valeurs_extremes_normales']['d_cap']})")
    print(f"  d_alt      alerte au-dessus de {s['d_alt']:.0f} ft/message   (vu en normal : max {a['valeurs_extremes_normales']['d_alt']})")
    if "cap_abs" in s:
        print(f"  cap_abs    alerte au-dessus de {s['cap_abs']:.2f}°   (vu en normal : max {a['valeurs_extremes_normales']['cap_abs']}°)")


# =====================================================================================================
# 3. Détection
# =====================================================================================================
def detecter(source, modele):
    """Retourne un DataFrame (un message par ligne) avec : alerte, score, motif, type_probable."""
    if isinstance(modele, str):
        modele = charger_modele(modele)
    s = modele["seuils"]
    f = source.copy() if isinstance(source, pd.DataFrame) and "pas_norm" in source else charger(source, modele["params"])
    ev = f.evalue
    seuil_cap_abs = s.get("cap_abs", np.inf)              # modèles v1 : pas de cap_abs

    saut = ev & (f.pas_norm > s["pas_norm_haut"])
    fige = ev & ((f.fige_run >= s["fige_run"]) | (f.pas_norm < s["pas_norm_bas"]))
    cap = ev & ((f.d_cap > s["d_cap"]) | (f.cap_abs > seuil_cap_abs))
    alt = ev & (f.d_alt > s["d_alt"])
    f["alerte"] = saut | fige | cap | alt
    precedent = f.groupby("id").alerte.shift(1) == True          # noqa: E712  (NaN -> False)
    f["debut_alerte"] = f.alerte & ~precedent                     # 1er message d'une série d'alertes
    f["serie_alerte"] = (f.alerte != precedent).groupby(f.id).cumsum()

    # score > 1  <=>  hors de l'enveloppe normale (le plus grand dépassement relatif parmi les variables)
    score = np.maximum.reduce([
        f.pas_norm / s["pas_norm_haut"],
        s["pas_norm_bas"] / np.maximum(f.pas_norm, 0.02),
        f.fige_run / s["fige_run"],
        f.d_cap / s["d_cap"],
        f.d_alt / s["d_alt"],
        f.cap_abs / seuil_cap_abs,
    ])
    f["score"] = np.where(ev, np.clip(score, 0, 50), 0.0)

    motif = pd.Series("", index=f.index)
    for masque, texte in [(saut, "saut de position"), (fige, "avion figé"), (cap, "cap incohérent"), (alt, "altitude brusque")]:
        motif = motif.where(~masque, (motif + "; " + texte).str.strip("; "))
    f["motif"] = motif
    f["type_probable"] = np.select(
        [saut | alt, fige, cap], ["Jumping aircraft", "Aircraft standing still", "Trajectory modification"], default="")
    f["prediction"] = np.where(f.alerte, f.type_probable, NORMAL)   # format commun avec le modèle supervisé
    return f


def resume_par_avion(res):
    """Un avion par ligne, les plus anormaux en premier."""
    lignes = []
    score = "score" if "score" in res else "proba_attaque"          # supervisé : probabilité d'attaque
    for i, a in res.groupby("id", sort=False):
        al = a[a.alerte]
        attaques = a.attackType[a.attackType != NORMAL]
        lignes.append({
            "id": i, "callsign": a.callsign.iloc[0], "messages": len(a), "en_vol": int(a.evalue.sum()),
            "alertes": len(al),
            "1re_alerte_msg": int(al.k.min()) if len(al) else None,
            "score_max": round(float(a[score].max()), 2),
            "type_probable": al.prediction.mode().iloc[0] if len(al) else "",
            "motifs": "; ".join(sorted(set("; ".join(al.motif).split("; ")) - {""})) if len(al) and "motif" in al else "",
            "etiquette": attaques.mode().iloc[0] if len(attaques) else (NORMAL if (a.attackType == NORMAL).any() else "-"),
        })
    r = pd.DataFrame(lignes).set_index("id")
    return r.sort_values(["alertes", "score_max"], ascending=False)


def evaluer(res, afficher=True):
    """Si le CSV contient des étiquettes : tableaux d'évaluation (voir evaluation_adsb.py)."""
    import evaluation_adsb as ev
    return ev.evaluer(res, "prediction", afficher=afficher)


# =====================================================================================================
# 4. Graphiques (palette bleu / orange ; forme du marqueur = type d'alerte)
# =====================================================================================================
SURFACE, TEXTE, TEXTE2, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLEU, ORANGE = "#2a78d6", "#eb6834"
FORMES = {"saut de position": "^", "avion figé": "s", "cap incohérent": "D", "altitude brusque": "o"}


def _style(ax, titre, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(titre, loc="left", fontsize=10.5, color=TEXTE)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXTE2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXTE2, fontsize=9)
    ax.grid(True, color=GRILLE, lw=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(GRILLE)
    ax.tick_params(colors=TEXTE2, labelsize=8.5)


def _marqueurs_alertes(ax, x, y, motifs):
    """Un marqueur au DÉBUT de chaque série d'alertes (forme = type d'alerte, couleur = alerte)."""
    for texte, forme in FORMES.items():
        m = motifs.str.contains(texte)
        if m.any():
            ax.scatter(x[m], y[m], marker=forme, s=46, color=ORANGE, edgecolor=SURFACE, linewidth=0.8,
                       zorder=5, label=f"début d'alerte : {texte}")


def _bandes_alertes(ax, a):
    """Zone orange pâle sur les messages en alerte (graphiques en fonction du n° de message)."""
    for _, serie in a[a.alerte].groupby("serie_alerte"):
        ax.axvspan(serie.k.min() - 0.5, serie.k.max() + 0.5, color=ORANGE, alpha=0.13, lw=0)


def tracer_vue_ensemble(res, sortie=None):
    """Carte des avions en vol avec le début de chaque série d'alertes, et nombre de messages en alerte par motif."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=SURFACE)
    vol = res[res.evalue] if res.evalue.any() else res
    lat0, lon0 = vol.latitude.median(), vol.longitude.median()
    res = res.assign(x_km=(res.longitude - lon0) * 111.32 * np.cos(np.radians(lat0)), y_km=(res.latitude - lat0) * 110.57)
    vol = res.loc[vol.index]
    # la capture peut contenir plusieurs aéroports (changement en cours de route) : on garde la zone principale
    proche = np.hypot(vol.x_km, vol.y_km) <= 800
    hors_zone = vol.id.nunique() - vol[proche].id.nunique()
    vol = vol[proche]
    for _, a in vol.groupby("id", sort=False):
        a1.plot(a.x_km, a.y_km, color=BLEU, lw=0.6, alpha=0.35)
    al = res[res.debut_alerte & res.index.isin(vol.index)]
    _marqueurs_alertes(a1, al.x_km, al.y_km, al.motif)
    a1.set_aspect("equal")
    marge = 5 + 0.05 * max(np.ptp(vol.x_km), np.ptp(vol.y_km))
    a1.set_xlim(vol.x_km.min() - marge, vol.x_km.max() + marge)
    a1.set_ylim(vol.y_km.min() - marge, vol.y_km.max() + marge)
    complement = f" ({hors_zone} avions d'une autre zone non affichés)" if hors_zone else ""
    _style(a1, f"Positions annoncées, {vol.id.nunique()} avions en vol{complement}", "km (nord)", "km (est)")
    if len(al):
        a1.legend(fontsize=8, frameon=False, loc="upper right")
    n = {t: int(res.motif.str.contains(t).sum()) for t in FORMES}
    a2.barh(list(n.keys()), list(n.values()), color=ORANGE, height=0.5)
    for y, v in enumerate(n.values()):
        a2.text(v, y, f" {v}", va="center", color=TEXTE, fontsize=9)
    a2.invert_yaxis()
    _style(a2, "Messages en alerte par motif")
    fig.suptitle("Anomalies détectées", x=0.01, ha="left", fontsize=13, color=TEXTE)
    fig.tight_layout()
    if sortie:
        fig.savefig(sortie, dpi=130, facecolor=SURFACE)
    return fig


def tracer_avion(res, ident, modele, contexte=60, sortie=None):
    """4 vues d'un avion : trajectoire, déplacement vs vitesse annoncée, dérive du cap, altitude."""
    import matplotlib.pyplot as plt
    if isinstance(modele, str):
        modele = charger_modele(modele)
    s = modele["seuils"]
    a = res[res.id == ident].copy()
    if a.empty:
        sys.exit(f"Avion inconnu : {ident}")
    if a.alerte.any():
        k_alertes = a[a.alerte].k
        a = a[(a.k >= k_alertes.min() - contexte) & (a.k <= k_alertes.max() + contexte)]
    a = a[a.evalue | a.alerte | (a.groundSpeed > 0)]
    lat0, lon0 = a.latitude.mean(), a.longitude.mean()
    a = a.assign(x_km=(a.longitude - lon0) * 111.32 * np.cos(np.radians(lat0)), y_km=(a.latitude - lat0) * 110.57)
    debuts = a[a.debut_alerte]
    etiq = ", ".join(sorted(set(a.attackType) - {NORMAL})) or "aucune étiquette d'attaque"

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    p = ax[0, 0]
    p.plot(a.x_km, a.y_km, color=BLEU, lw=1.5, label="positions annoncées")
    p.plot(a.x_km.iloc[0], a.y_km.iloc[0], "o", color=TEXTE2, ms=5, label="premier message affiché")
    _marqueurs_alertes(p, debuts.x_km, debuts.y_km, debuts.motif)
    p.set_aspect("equal", adjustable="datalim")
    _style(p, "Trajectoire annoncée", "km (nord)", "km (est)")
    p.legend(fontsize=8, frameon=False, loc="best")

    p = ax[0, 1]
    p.axhspan(s["pas_norm_bas"], s["pas_norm_haut"], color=BLEU, alpha=0.10, label="enveloppe normale apprise")
    _bandes_alertes(p, a)
    p.plot(a.k, a.pas_norm.where(a.evalue), color=BLEU, lw=1.5)
    _marqueurs_alertes(p, debuts.k, debuts.pas_norm, debuts.motif)
    if a.pas_norm.max() > 20:
        p.set_yscale("symlog", linthresh=2)
    _style(p, "Déplacement réel / déplacement attendu (1 = normal)", None, "n° de message")
    p.legend(fontsize=8, frameon=False)

    p = ax[1, 0]
    _bandes_alertes(p, a)
    p.plot(a.k, a.cap_abs.where(a.evalue), color=BLEU, lw=1.5)
    if "cap_abs" in s:
        p.axhline(s["cap_abs"], color=TEXTE2, lw=1, ls="--", label=f"seuil appris = {s['cap_abs']:.1f}°")
    _marqueurs_alertes(p, debuts.k, debuts.cap_abs, debuts.motif)
    _style(p, "Écart entre cap annoncé et trajectoire réelle (°)", None, "n° de message")
    p.legend(fontsize=8, frameon=False)

    p = ax[1, 1]
    _bandes_alertes(p, a)
    p.plot(a.k, a.altitude, color=BLEU, lw=1.5)
    _marqueurs_alertes(p, debuts.k, debuts.altitude, debuts.motif)
    _style(p, "Altitude annoncée (ft)", None, "n° de message")

    fig.suptitle(f"{a.callsign.iloc[0]} ({ident}) ; étiquette du simulateur : {etiq}", x=0.01, ha="left", fontsize=13, color=TEXTE)
    fig.tight_layout()
    if sortie:
        fig.savefig(sortie, dpi=130, facecolor=SURFACE)
    return fig


# =====================================================================================================
# 5. Ligne de commande
# =====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Détecteur d'anomalies ADS-B (OpenScope-sec)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("entrainer", help="apprendre le trafic normal")
    e.add_argument("csv", nargs="+", help="une ou plusieurs captures (.csv ou .csv.gz)")
    e.add_argument("--modele", default="modeles/modele_normal.json")
    d = sp.add_parser("detecter", help="détecter les anomalies dans une capture")
    d.add_argument("csv")
    d.add_argument("--modele", default="modeles/modele_normal.json")
    d.add_argument("--id", help="avion à tracer (défaut : le plus anormal)")
    d.add_argument("--sortie", default="rapport", help="préfixe des images produites")
    args = ap.parse_args()

    if args.cmd == "entrainer":
        afficher_modele(entrainer(args.csv, args.modele))
        return

    import matplotlib
    matplotlib.use("Agg")
    modele = charger_modele(args.modele)
    res = detecter(args.csv, modele)
    resume = resume_par_avion(res)
    print(f"\n{res.id.nunique()} avions, {len(res)} messages, {int(res.alerte.sum())} messages en alerte, "
          f"{int((resume.alertes > 0).sum())} avions concernés\n")
    print(resume[resume.alertes > 0].head(15).to_string())
    if (res.attackType != NORMAL).any():
        print("\n=== Comparaison avec les étiquettes du simulateur ===")
        evaluer(res)
    tracer_vue_ensemble(res, f"{args.sortie}_ensemble.png")
    cible = args.id or (resume.index[0] if resume.alertes.iloc[0] > 0 else None)
    if cible:
        tracer_avion(res, cible, modele, sortie=f"{args.sortie}_{cible}.png")
        print(f"\nImages : {args.sortie}_ensemble.png et {args.sortie}_{cible}.png")
    res[res.alerte].to_csv(f"{args.sortie}_alertes.csv", index=False)


if __name__ == "__main__":
    main()
