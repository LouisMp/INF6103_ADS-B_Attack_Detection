# Résultats d'évaluation

Généré par `python experiences.py`. Définition de la vérité terrain : voir `evaluation_adsb.py`.

## Données

| jeu | messages | avions | messages en vol évalués | vérité Normal | vérité Jumping | vérité Standing still | vérité VTM | vérité zone grise |
|---|---|---|---|---|---|---|---|---|
| A + B1 (entraînement / validation croisée) | 354911 | 591 | 281327 | 195037 | 295 | 2647 | 28934 | 54414 |
| B0 (test) | 152895 | 301 | 129271 | 11708 | 251 | 6189 | 23679 | 87444 |

## 1. Validation croisée groupée par avion (A + B1)

### Tableau comparatif

| modèle | attaque | précision | rappel | F1 | avions détectés (effet) | avions détectés (étiquette brute) | retard médian (msgs) | fausses alertes / 1000 msgs normaux | avions normaux avec fausse alerte |
|---|---|---|---|---|---|---|---|---|---|
| Non supervisé (enveloppe normale) | Jumping | 1.000 | 1.000 | 1.000 | 19/19 | 19/58 | 0.000 |  |  |
| Non supervisé (enveloppe normale) | Standing still | 1.000 | 1.000 | 1.000 | 14/14 | 14/39 | 0.000 |  |  |
| Non supervisé (enveloppe normale) | VTM | 1.000 | 0.988 | 0.994 | 37/37 | 37/37 | 5.000 |  |  |
| Non supervisé (enveloppe normale) | Toute attaque (binaire) | 1.000 | 0.989 | 0.994 |  |  |  | 0.000 | 0/346 |
| Supervisé (forêt aléatoire) | Jumping | 1.000 | 1.000 | 1.000 | 19/19 | 19/58 | 0.000 |  |  |
| Supervisé (forêt aléatoire) | Standing still | 1.000 | 1.000 | 1.000 | 14/14 | 14/39 | 0.000 |  |  |
| Supervisé (forêt aléatoire) | VTM | 1.000 | 0.995 | 0.998 | 37/37 | 37/37 | 1.000 |  |  |
| Supervisé (forêt aléatoire) | Toute attaque (binaire) | 1.000 | 0.996 | 0.998 |  |  |  | 0.005 | 1/346 |

### Non supervisé (enveloppe normale)

Par message (zone grise exclue) :

| classe | messages_attaqués | VP | FP | FN | précision | rappel | F1 | rappel_toute_alerte |
|---|---|---|---|---|---|---|---|---|
| Jumping | 295 | 295 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Standing still | 2647 | 2647 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| VTM | 28934 | 28575 | 0 | 359 | 1.000 | 0.988 | 0.994 | 0.988 |
| Toute attaque (binaire) | 31876 | 31517 | 0 | 359 | 1.000 | 0.989 | 0.994 | 0.989 |

Fausses alertes : **0** sur 195037 messages normaux (**0.00 pour 1000**).

Par avion :

| attaque | avions_étiquetés_en_vol | détectés_parmi_étiquetés | taux_étiquetés | avions_avec_effet | détectés_parmi_effet | taux_effet | bon_type | retard_médian_msgs | retard_max_msgs |
|---|---|---|---|---|---|---|---|---|---|
| Jumping | 58 | 19 | 0.328 | 19.0 | 19.0 | 1.000 | 19.0 | 0.000 | 0.000 |
| Standing still | 39 | 14 | 0.359 | 14.0 | 14.0 | 1.000 | 14.0 | 0.000 | 0.000 |
| VTM | 37 | 37 | 1.000 | 37.0 | 37.0 | 1.000 | 37.0 | 5.000 | 10.0 |
| Normal (jamais attaqué) | 346 | 0 | 0.000 |  |  |  |  |  |  |

Matrice de confusion (messages ; lignes = vérité, colonnes = prédiction) :

| vérité | Normal | Jumping | Standing still | VTM |
|---|---|---|---|---|
| Normal | 195037 | 0 | 0 | 0 |
| Jumping | 0 | 295 | 0 | 0 |
| Standing still | 0 | 0 | 2647 | 0 |
| VTM | 359 | 0 | 0 | 28575 |

### Supervisé (forêt aléatoire)

Par message (zone grise exclue) :

| classe | messages_attaqués | VP | FP | FN | précision | rappel | F1 | rappel_toute_alerte |
|---|---|---|---|---|---|---|---|---|
| Jumping | 295 | 295 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Standing still | 2647 | 2647 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| VTM | 28934 | 28793 | 1 | 141 | 1.000 | 0.995 | 0.998 | 0.995 |
| Toute attaque (binaire) | 31876 | 31735 | 1 | 141 | 1.000 | 0.996 | 0.998 | 0.996 |

Fausses alertes : **1** sur 195037 messages normaux (**0.01 pour 1000**).

Par avion :

| attaque | avions_étiquetés_en_vol | détectés_parmi_étiquetés | taux_étiquetés | avions_avec_effet | détectés_parmi_effet | taux_effet | bon_type | retard_médian_msgs | retard_max_msgs |
|---|---|---|---|---|---|---|---|---|---|
| Jumping | 58 | 19 | 0.328 | 19.0 | 19.0 | 1.000 | 19.0 | 0.000 | 0.000 |
| Standing still | 39 | 14 | 0.359 | 14.0 | 14.0 | 1.000 | 14.0 | 0.000 | 0.000 |
| VTM | 37 | 37 | 1.000 | 37.0 | 37.0 | 1.000 | 37.0 | 1.000 | 5.000 |
| Normal (jamais attaqué) | 346 | 1 | 0.003 |  |  |  |  |  |  |

Matrice de confusion (messages ; lignes = vérité, colonnes = prédiction) :

| vérité | Normal | Jumping | Standing still | VTM |
|---|---|---|---|---|
| Normal | 195036 | 0 | 0 | 1 |
| Jumping | 0 | 295 | 0 | 0 |
| Standing still | 0 | 0 | 2647 | 0 |
| VTM | 141 | 0 | 0 | 28793 |

## 2. Test indépendant (B0)

### Tableau comparatif

| modèle | attaque | précision | rappel | F1 | avions détectés (effet) | avions détectés (étiquette brute) | retard médian (msgs) | fausses alertes / 1000 msgs normaux | avions normaux avec fausse alerte |
|---|---|---|---|---|---|---|---|---|---|
| Non supervisé (enveloppe normale) | Jumping | 0.940 | 1.000 | 0.969 | 51/51 | 92/203 | 0.000 |  |  |
| Non supervisé (enveloppe normale) | Standing still | 0.976 | 1.000 | 0.988 | 22/22 | 22/60 | 0.000 |  |  |
| Non supervisé (enveloppe normale) | VTM | 1.000 | 0.939 | 0.969 | 83/83 | 83/83 | 4.000 |  |  |
| Non supervisé (enveloppe normale) | Toute attaque (binaire) | 0.999 | 0.957 | 0.977 |  |  |  | 2.733 | 2/27 |
| Supervisé (forêt aléatoire) | Jumping | 0.940 | 1.000 | 0.969 | 51/51 | 89/203 | 0.000 |  |  |
| Supervisé (forêt aléatoire) | Standing still | 0.976 | 1.000 | 0.988 | 22/22 | 22/60 | 0.000 |  |  |
| Supervisé (forêt aléatoire) | VTM | 1.000 | 0.967 | 0.983 | 83/83 | 83/83 | 2.000 |  |  |
| Supervisé (forêt aléatoire) | Toute attaque (binaire) | 0.999 | 0.979 | 0.989 |  |  |  | 2.733 | 2/27 |

### Non supervisé (enveloppe normale)

Par message (zone grise exclue) :

| classe | messages_attaqués | VP | FP | FN | précision | rappel | F1 | rappel_toute_alerte |
|---|---|---|---|---|---|---|---|---|
| Jumping | 251 | 251 | 16 | 0 | 0.940 | 1.000 | 0.969 | 1.000 |
| Standing still | 6189 | 6189 | 152 | 0 | 0.976 | 1.000 | 0.988 | 1.000 |
| VTM | 23679 | 22237 | 0 | 1442 | 1.000 | 0.939 | 0.969 | 0.945 |
| Toute attaque (binaire) | 30119 | 28813 | 32 | 1306 | 0.999 | 0.957 | 0.977 | 0.957 |

Fausses alertes : **32** sur 11708 messages normaux (**2.73 pour 1000**).

Par avion :

| attaque | avions_étiquetés_en_vol | détectés_parmi_étiquetés | taux_étiquetés | avions_avec_effet | détectés_parmi_effet | taux_effet | bon_type | retard_médian_msgs | retard_max_msgs |
|---|---|---|---|---|---|---|---|---|---|
| Jumping | 203 | 92 | 0.453 | 51.0 | 51.0 | 1.000 | 51.0 | 0.000 | 0.000 |
| Standing still | 60 | 22 | 0.367 | 22.0 | 22.0 | 1.000 | 22.0 | 0.000 | 0.000 |
| VTM | 83 | 83 | 1.000 | 83.0 | 83.0 | 1.000 | 76.0 | 4.000 | 10.0 |
| Normal (jamais attaqué) | 27 | 2 | 0.074 |  |  |  |  |  |  |

Matrice de confusion (messages ; lignes = vérité, colonnes = prédiction) :

| vérité | Normal | Jumping | Standing still | VTM |
|---|---|---|---|---|
| Normal | 11676 | 0 | 32 | 0 |
| Jumping | 0 | 251 | 0 | 0 |
| Standing still | 0 | 0 | 6189 | 0 |
| VTM | 1306 | 16 | 120 | 22237 |

### Supervisé (forêt aléatoire)

Par message (zone grise exclue) :

| classe | messages_attaqués | VP | FP | FN | précision | rappel | F1 | rappel_toute_alerte |
|---|---|---|---|---|---|---|---|---|
| Jumping | 251 | 251 | 16 | 0 | 0.940 | 1.000 | 0.969 | 1.000 |
| Standing still | 6189 | 6189 | 151 | 0 | 0.976 | 1.000 | 0.988 | 1.000 |
| VTM | 23679 | 22902 | 0 | 777 | 1.000 | 0.967 | 0.983 | 0.973 |
| Toute attaque (binaire) | 30119 | 29477 | 32 | 642 | 0.999 | 0.979 | 0.989 | 0.979 |

Fausses alertes : **32** sur 11708 messages normaux (**2.73 pour 1000**).

Par avion :

| attaque | avions_étiquetés_en_vol | détectés_parmi_étiquetés | taux_étiquetés | avions_avec_effet | détectés_parmi_effet | taux_effet | bon_type | retard_médian_msgs | retard_max_msgs |
|---|---|---|---|---|---|---|---|---|---|
| Jumping | 203 | 89 | 0.438 | 51.0 | 51.0 | 1.000 | 51.0 | 0.000 | 0.000 |
| Standing still | 60 | 22 | 0.367 | 22.0 | 22.0 | 1.000 | 22.0 | 0.000 | 0.000 |
| VTM | 83 | 83 | 1.000 | 83.0 | 83.0 | 1.000 | 76.0 | 2.000 | 5.000 |
| Normal (jamais attaqué) | 27 | 2 | 0.074 |  |  |  |  |  |  |

Matrice de confusion (messages ; lignes = vérité, colonnes = prédiction) :

| vérité | Normal | Jumping | Standing still | VTM |
|---|---|---|---|---|
| Normal | 11676 | 0 | 32 | 0 |
| Jumping | 0 | 251 | 0 | 0 |
| Standing still | 0 | 0 | 6189 | 0 |
| VTM | 642 | 16 | 119 | 22902 |

## Importance des variables (forêt aléatoire finale)

| variable | importance |
|---|---|
| pas_norm | 0.341 |
| cap_abs | 0.323 |
| fige_run | 0.185 |
| d_alt | 0.104 |
| d_cap | 0.047 |

## Seuils appris (non supervisé final)

```json
{
  "seuils": {
    "pas_norm_bas": 0.8,
    "pas_norm_haut": 1.2,
    "fige_run": 3,
    "d_cap": 0.545,
    "d_alt": 100,
    "cap_abs": 2.996
  },
  "apprentissage": {
    "captures": [
      "capture_A_normal",
      "capture_B1_entrainement"
    ],
    "avions": 346,
    "messages": 195037,
    "avions_ecartes": [],
    "valeurs_extremes_normales": {
      "pas_norm": [
        0.927,
        1.0545
      ],
      "fige_run": 0.0,
      "d_cap": 0.3633,
      "d_alt": 27.22,
      "cap_abs": 1.997
    }
  }
}
```

