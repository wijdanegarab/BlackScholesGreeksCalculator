# Option Pricing Engine

## Pourquoi ce projet ?

J'ai voulu comprendre comment évaluer une option financière en implémentant deux approches complètement différentes : la formule analytique de Black-Scholes et la simulation Monte Carlo. C'était important pour moi de valider que les deux méthodes convergent vers le même résultat.

## Ce qu'il fait

Le projet propose deux façons d'évaluer des options européennes :

**Black-Scholes** - Une formule mathématique directe qui donne le prix en quelques millisecondes.

**Monte Carlo** - Une simulation qui crée 50 000 futurs possibles du prix de l'action pour estimer la valeur moyenne de l'option.

J'ai aussi implémenté les **Greeks** (Delta, Gamma, Vega, Theta, Rho) qui mesurent les risques associés à l'option.

## Installation et utilisation
```bash
pip install numpy scipy matplotlib
python main.py
```

Les librairies utilisées sont basiques : NumPy pour les calculs, SciPy pour les distributions statistiques, Matplotlib pour les graphiques.

## Comment ça marche

### Black-Scholes

La formule est simple une fois qu'on la comprend :
```
C = S·N(d1) - K·e^(-rT)·N(d2)
```

C'est l'espérance actualisée du payoff. Je calcule d1 et d2 selon les paramètres du marché, puis j'utilise la distribution normale cumulée N() pour obtenir le prix.

### Monte Carlo

L'idée est différente : au lieu de chercher une formule, je **simule** le futur.

1. Le prix suit un mouvement Brownien géométrique : `dS = μS dt + σS dW`
2. Je discrétise ça avec Euler : chaque jour, le prix change par une petite tendance + du bruit aléatoire
3. Je crée 50 000 scénarios possibles de 100 jours
4. Pour chaque scénario, je calcule le gain à l'expiration
5. J'en fais la moyenne et j'actualise

C'est plus lent que Black-Scholes (quelques secondes vs 1 milliseconde), mais ça **marche pour n'importe quel type d'option**. 

## Validation

La vraie question était : est-ce que mon Monte Carlo est correct ?

J'ai comparé avec Black-Scholes. Avec 50 000 simulations, l'erreur est **< 0.15%**. C'est bon signe que l'implémentation est juste.

Les graphiques de convergence le montrent bien : plus on augmente le nombre de simulations, plus MC se rapproche de BS.

## Visualisations

J'ai généré 6 graphiques pour montrer différents aspects :

- **Delta** - Comment le prix de l'option change quand le stock change
- **CALL vs PUT** - Les deux types d'options (call = achat, put = vente)
- **Time Decay** - Pourquoi une option perd de la valeur en se rapprochant de l'expiration
- **MC Convergence** - Preuve que l'algorithme fonctionne
- **Simulated Paths** - À quoi ressemblent les futurs possibles
- **Price Distribution** - Statistiques des prix finaux

## Structure du code

- `option.py` - La classe qui calcule Black-Scholes
- `monte_carlo.py` - La simulation Monte Carlo
- `main.py` - Génère tous les graphiques et la comparaison

J'ai séparé les deux approches en classes différentes pour que ce soit clair que ce sont deux façons indépendantes de résoudre le même problème.


## Notes

Les paramètres par défaut simulent une option Apple CALL sur 3 mois avec 150€ spot et 155€ strike. Tu peux les modifier pour tester différents scénarios.
