##  Analyse des résultats du modèle TFT (minute) – Dropout 0.25, 50 epochs

### **1. Contexte et amélioration obtenue**

Les premiers entraînements sur la série temporelle à résolution *minute* donnaient des résultats insatisfaisants :

* **MAE > 2000**, **RMSE ≈ 4000**, **R² < 0**.
* Le **QuantileLoss** (perte utilisée pour entraîner le modèle) stagnait autour de **1200 à 1500**, sans convergence significative.
* Or, les valeurs typiques de consommation sont comprises entre **50 000 et 60 000**, avec des pointes > 90 000. Une erreur de 1000 représente déjà un écart faible relativement à cette échelle.

La nouvelle version (Temporal Fusion Transformer + Dropout 0.25 + encodage 6h) marque une **amélioration nette** sur tous les plans :

---

### **2. Paramètres du modèle**

| Paramètre                | Valeur         |
| ------------------------ | -------------- |
| `max_encoder_length`     | 288 (6 × 48)   |
| `max_prediction_length`  | 48             |
| `dropout`                | 0.25           |
| `batch_size`             | 128            |
| `learning_rate`          | 0.005          |
| `hidden_size`            | 256            |
| `attention_head_size`    | 8              |
| `hidden_continuous_size` | 64             |
| `max_epochs`             | 50             |
| `accelerator`            | GPU (1 device) |

**Variables :**

* `known_reals`: `["time_idx", "apparent_temperature", "is_vacation"]`
* `unknown_reals`: `["Consommation", "sunshine_duration"]`
* `known_categoricals`: `["weekday", "month", "season"]`
* `columns_to_drop`: `["Prévision J", "Prévision J-1"]`

---

### **3. Résultats obtenus**

| Métrique                       | Valeur                   |
| ------------------------------ | ------------------------ |
| **MAE**                        | **991.44**               |
| **RMSE**                       | **1413.38**              |
| **R²**                         | **0.83**                 |
| **Couverture P5–P95**          | **64.58%**               |
| **Train Loss (step)**          | 150                      |
| **Train Loss (epoch)**         | 267                      |
| **Validation Loss (epoch)**    | 725                      |
| **Min Validation Loss (step)** | **474** @ step **22112** |

---

### **4. Interprétation avancée**

####  *QuantileLoss et couverture*

Le modèle a été entraîné avec une **QuantileLoss** (`QuantileLoss(quantiles=[0.05, ..., 0.95])`) avec `output_size=7`. Cela signifie qu’il apprend à prédire plusieurs quantiles, en particulier le 0.05 (p5), le 0.5 (médiane), et le 0.95 (p95).

Le **QuantileLoss** correspond ici à la moyenne des pertes sur ces quantiles — il pénalise les sous- ou sur-prédictions selon leur quantile cible. Une perte faible implique que les quantiles sont correctement prédits.

#### *Améliorations majeures*

* **Le R² est passé de < 0 à 0.83**, confirmant une bien meilleure captation de la variance.
* **MAE** et **RMSE** ont été réduits de plus de **50%**, malgré une échelle absolue élevée.
* La **validation loss minimale (474)** a été atteinte vers le **step 22 112**, indiquant une phase d’optimisation efficace.
* Le **gap entre train et validation loss (> 200)** montre un début de surapprentissage après \~20 epochs.

####  *Sur la couverture (64.58%)*

C’est un point faible. L’intervalle \[P5–P95] ne couvre que \~65% des vraies valeurs. Cela signifie que les intervalles prédits sont trop étroits ou mal calibrés. Or, pour un usage industriel ou opérationnel, on préférerait **> 90% de couverture**.

### **5. Recommandations pour la suite**

| Action à mener                                     | Objectif principal                                         |
| -------------------------------------------------- | ---------------------------------------------------------- |
| **Identifier le step avec le plus bas val\_loss**  | Évaluer le point d'arrêt idéal                             |
| **Mettre en place un early stopping**              | Stopper avant surapprentissage                             |
| **Activer les checkpoints**                        | Sauvegarder automatiquement les meilleurs poids            |
| **Augmenter le dropout à 0.30**                    | Renforcer la régularisation                                |
| **Réduire `max_epochs` à 20 ou 30**                | Éviter perte de temps et surapprentissage                  |
| **Passer à `QuantileLoss(quantiles=[0.95])` seul** | Focus sur borne supérieure pour garantie de dépassement    |
| **Formuler une promesse sur le P95**               | Ex: “dans 95% des cas, la consommation ne dépassera pas Y” |


###  Analyse de l’impact du `dropout` sur la stabilité de l’apprentissage (comparaison des essais)

Dans cette nouvelle itération, l’entraînement a été lancé avec un `dropout` fixé à **0.30** (contre 0.25 dans l’essai précédent) et avec un mécanisme d’**early stopping** (patience de 5). L’objectif était de limiter le surapprentissage tout en observant si un taux de régularisation plus élevé permettrait de stabiliser davantage la validation loss.

####  Résultats obtenus :

* **Nombre d’époques entraînées** : 10 (arrêt automatique via early stopping).
* **Train loss (epoch)** : 908
* **Val loss (final)** : 1080
* **MAE** : 1983.72
* **RMSE** : 2463.33
* **R²** : 0.49
* **Coverage (entre 5% et 95%)** : 87.5 %

####  Interprétation :

Comparé à l’essai précédent (`dropout=0.25`, sans early stopping), le modèle montre ici :

* une **validation loss plus élevée** (≈1080 contre ≈600 auparavant),
* un **R² nettement plus faible** (0.49 contre 0.83),
* et une **convergence plus lente**, malgré des paramètres similaires sur le reste de l’architecture.

Ces observations suggèrent que le `dropout=0.30` était **trop contraignant** pour le modèle : il a rendu l’apprentissage instable, empêchant le réseau de capter correctement les motifs sous-jacents, surtout en début d’entraînement. Cela s’est traduit par une incapacité à atteindre les performances obtenues précédemment, même avec plus d’itérations.

#### Conclusion :

* Un **dropout de 0.30** semble trop élevé pour cette tâche.
* La configuration précédente (`dropout=0.25`) offrait un **meilleur équilibre entre régularisation et capacité d’apprentissage**.
* Il est recommandé de conserver `dropout=0.25`, tout en ajustant si besoin le `early_stopping` (par exemple : patience=5) pour contrôler les surapprentissages éventuels.


