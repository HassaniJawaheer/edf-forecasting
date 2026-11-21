### Tableau récapitulatif des dépendances nécessaires

| **Dépendance** | **Version recommandée** | **Pourquoi** | **Commentaire** |
|:---------------|:------------------------|:-------------|:----------------|
| Python | 3.10 | Base système pour compatibilité packages | (`>=3.8,<3.11` requis par Forecasting 1.0.0) |
| PyTorch | 2.0.1+cu118 | Backend deep learning | Version GPU adaptée pour RTX 4070 |
| PyTorch Lightning | 2.0.2 | Framework training | Compatible avec PyTorch 2.0.1 |
| PyTorch Forecasting | 1.0.0 | Modèle TFT | Contient TemporalFusionTransformer et TimeSeriesDataSet |
| pandas | latest stable | Manipulation des DataFrames | |
| numpy | latest stable | Calcul scientifique | |
| matplotlib | latest stable | Visualisation de courbes | |
| scikit-learn | latest stable | Support baseline model, métriques | |
| optuna | latest stable (optionnel) | Tuning hyperparamètres | Seulement pour phase avancée |
| tensorboard | latest stable (optionnel) | Visualisation logs training | Optionnel mais utile pour suivi |

---

### 📦 Résumé en format commandes `pip`

En environnement Python 3.10 :

```bash
pip install torch==2.0.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.0.2
pip install git+https://github.com/jdb78/pytorch-forecasting.git@v1.0.0
pip install pandas numpy matplotlib scikit-learn
pip install optuna tensorboard  # (optionnel, conseillé pour tuning et visualisation)
```

---

### ⚠️ Détail important sur l’installation :

- **pytorch-forecasting** 1.0.0 **n’est pas disponible sur PyPI**, il faut passer par GitHub avec `git+https`.
- Si tu as **pas de GPU**, on utilisera **torch==2.0.1+cpu** au lieu de `+cu118`.
- Pour CUDA (`+cu118`), tu es tranquille avec ta RTX 4070.

---

### 🧠 Remarques stratégiques :

| Point | Décision |
|:------|:---------|
| Python 3.10 obligatoire | Oui, impératif pour éviter conflit de versions |
| Baseline modèle (Baseline() predict) | Scikit-learn suffit, pas besoin d’outils exotiques |
| Tuning hyperparamètres | Optuna sera utilisé seulement si tu veux optimiser après premiers tests |
| Monitoring entrainement | Tensorboard activable mais pas obligatoire pour le premier jet |

## Installation commands
```bash
docker build -t tft-env .
# Sans utilisr les caches
docker build --no-cache -t tft-env .
```
## Installation de nvidia-container-runtime
Important pour pouvoir que Docker puissent utiliser le GPU.
```bash
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
# Pour tester si tous est OK
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```
## Supprimer l'ancienne image Docker
```bash
docker rmi tft-env
```
## Premier test à l'aveugle sans donner de validation

### **Contexte du test**

- Objectif : **Premier entraînement rapide** du modèle **Temporal Fusion Transformer** (**TFT**) pour **tester la chaîne complète** de traitement (**Data ➔ Model ➔ Training**) sans validation.
- Hypothèse : **aucun set de validation** utilisé pour simplifier la boucle d’entraînement.  
- Dataset utilisé : **1 seule série temporelle**, préformatée avec :
  - `time_idx` = indice temporel (0 → N),
  - `Consommation` = variable cible,
  - `series_id` = identifiant de série constant (France).

---

### **Configuration du modèle Temporal Fusion Transformer**

- **Longueur de la séquence passée (encoder)** : `max_encoder_length = 48`
- **Horizon de prédiction (decoder)** : `max_prediction_length = 48`
- **Batch size** : `64`
- **Optimisation** :
  - **Optimiseur** : par défaut (Adam via PyTorch Lightning).
  - **Learning rate** : `0.001`
- **Architecture du réseau** :
  - **hidden_size** : `16` (taille interne LSTM + réseaux résiduels)
  - **attention_head_size** : `1` (une tête d'attention, modèle léger)
  - **hidden_continuous_size** : `8` (dimension pour variables continues)
  - **dropout** : `0.1`
  - **Output size** : `7` (prédictions sur 7 quantiles : `[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]`)
- **Fonction de coût (loss)** :
  - **QuantileLoss** (calcul sur plusieurs quantiles, principalement le médian (`q=0.5`))
  - Le `train_loss` rapporté correspond donc **à une erreur quantile médiane**, assimilable à **l'erreur absolue moyenne**.

---

### **État de l'entraînement**

- **Nombre d’époques** : `5`
- **Utilisation GPU** : `NVIDIA GeForce RTX 4070 Laptop GPU`
- **Gradient clipping** : `0.1` (stabilisation de l'entraînement)
- **Temps d'entraînement** : **environ 2 minutes**
- **Nombre total de paramètres du modèle** : **18.8k** (~0.075 MB)

---

### **Résultats observés**

- **Train Loss (médian) après 5 époques** : **environ 1260**
- **Interprétation de la valeur du loss** :
  - Cela signifie que **l'erreur moyenne par prédiction** est d’environ **1260 unités** (par exemple 1260 kWh si consommation énergétique).
  - Rapporté aux valeurs moyennes observées (typiquement entre **10 000** et **50 000**), cela représente une **erreur relative** comprise entre **2% et 12%** selon la magnitude de la série temporelle au moment de la prédiction.
- **Conclusion** :  
  - Cette performance est **tout à fait acceptable** pour un premier jet, **sans tuning d'hyperparamètres**, **sans validation**, **sans feature engineering**.
  - Le modèle **apprend correctement** la tendance générale de la série temporelle.

---

### **Remarques sur les avertissements**

| Type de Warning | Signification | Gravité |
|:----------------|:---------------|:--------|
| `loss` sauvegardé | Attribut `loss` enregistré deux fois |  Pas grave |
| `logging_metrics` sauvegardé | Attribut `metrics` enregistré deux fois | Pas grave |
| `val_dataloader absent` | Pas de validation configurée | Normal ici |
| `ReduceLROnPlateau` non activé | Pas de validation donc pas de réduction du LR | Normal ici |
| `torch.set_float32_matmul_precision` | Conseillé pour accélérer sur GPU (optionnel) | Optionnel |
| `DataLoader num_workers=0` | Peut ralentir un peu l'entraînement | Optimisable |

**Donc : aucun problème bloquant détecté.**

---

### **Points importants à noter pour la suite**

- Ajouter un vrai `validation_dataloader` pour pouvoir suivre le `val_loss` (pas uniquement `train_loss`).
- Possibilité d'utiliser **torch.set_float32_matmul_precision('medium')** au tout début pour exploiter mieux les Tensor Cores de ton GPU (optionnel pour l'instant).
- Tuning futur possible : augmenter `hidden_size`, `dropout`, ou `attention_head_size` si on veut un modèle plus robuste.
## Deuxième Test — Training + Validation sans tuning fin

### 1. **Les warnings qui apparaissent :**

- **`loss` et `logging_metrics` warning**  
  → Juste un rappel : Lightning trouve que dans ton modèle (`tft`), l'attribut `loss` et `logging_metrics` sont déjà des modules `nn.Module`, et donc il recommande *proprement* de les ignorer lors du `save_hyperparameters()`.  
  ➔ **Pas grave**, aucune conséquence pour l'entraînement. C’est juste une alerte pour des futurs développements plus propres.

- **Plain `ModelCheckpoint` au lieu de `LitModelCheckpoint`**  
  → Là aussi c’est une *amélioration optionnelle*. Tu utilises l'ancien système pour sauvegarder les modèles, il te propose de passer à `LitModelCheckpoint` qui est plus récent.  
  ➔ **Pas bloquant**, totalement **optionnel**.

- **CUDA Tensor Cores available warning**  
  → Ton GPU (`RTX 4070 Laptop GPU`) **a des Tensor Cores** rapides. Lightning recommande d'**activer `torch.set_float32_matmul_precision('medium')`** pour optimiser la vitesse d'entraînement sans perdre beaucoup de précision.  
  ➔ Tu pourrais l’ajouter au tout début de ton script si tu veux (je peux te le montrer). Encore une fois, **pas bloquant**, **juste optimisation**.

- **DataLoader 'num_workers' warning**  
  → Tu utilises `num_workers=0` dans ton `train_dataloader`. Ils te suggèrent de **mettre plus de workers** (genre 4, 8 ou même 15) pour **accélérer le chargement des données**.  
  ➔ Encore une fois, c’est **un conseil**, **pas une erreur**.

---

### 2. **Structure du modèle TemporalFusionTransformer**

- Tu as environ **18.8K paramètres entraînables** → Très léger, c’est **parfait pour du test rapide**.  
- Les modules utilisés :
  - **Embeddings** (`MultiEmbedding`) → Sert à encoder les variables discrètes/catégorielles (ici peu utilisé).
  - **Variable Selection Networks** → Ces modules apprennent à choisir **quelles variables temporelles ou statiques** sont importantes à chaque instant.
  - **Gated Residual Networks (GRN)** → Petits réseaux à portes utilisés pour enrichir l'information (encodage enrichi du contexte).
  - **LSTM Encoder / Decoder** → Gèrent la dynamique temporelle de la séquence.
  - **Interpretable Multi-Head Attention** → Permet de capter les dépendances temporelles longues (attention entre les pas de temps).
  - **Linear Output Layer** → Dernière couche pour prédire les quantiles demandés.

**Conclusion sur l'architecture** :  
➡️ C'est un modèle **complet et efficace**, bien conçu même si ici **avec peu de hidden_size (16)**, c'est normal que le modèle soit "compact".

---

### 3. **État de l'entraînement actuel**

- **Train Loss** autour de **1500**  
- **Validation Loss** autour de **2000-2500**  
- **Pas de surapprentissage violent** pour le moment, l'écart entre train et val n'est **pas dramatique** (surtout avec un dataset simple).

Donc ➔ **c’est sain pour un premier test**, tu peux continuer !

---

### 4. **Petit rappel sur l'interprétation du Loss ici**

Ton loss est un **Quantile Loss** :
- Ce n’est pas un simple RMSE ou MAE.
- Ça mesure l'écart **pour plusieurs quantiles** (ex: prédire correctement le 10%, 50%, 90%).
- **Valeur brute** difficile à interpréter seule.
- Il faut surtout **comparer** Loss entraînement vs validation (**pas juste sa valeur**).

---

## ➔ **Résumons ultra simplement**

| Élément                  | Statut         |
|---------------------------|----------------|
| Loss / Metrics Warnings    | ➔ Pas grave    |
| Tensor Core Warning        | ➔ Pas grave (optimisation possible) |
| DataLoader Workers         | ➔ Pas grave (optimisation possible) |
| Surapprentissage           | ➔ Non détecté pour l’instant |
| Modèle                      | ➔ Structure OK |
 
### Aprés entrainement
Après entraînement du modèle **Temporal Fusion Transformer** (TFT) sur la base préparée :

- **Paramètres de fenêtre** :  
  - Longueur d’encodage (`max_encoder_length`) : 48 pas
  - Longueur de prédiction (`max_prediction_length`) : 48 pas
- **Nombre d’époques** : 10
- **Batch size** : 64
- **Optimiseur utilisé** : learning rate fixé à 0.001, pas d'early stopping.

**Résultats d'entraînement** :
- **Train loss (epoch step)** : environ **1250**
- **Train loss (fin d'entraînement)** : environ **1270**
- **Validation loss** : environ **2230**

**Évaluation finale sur validation (à l'issue des 10 époques)** :
- **MAE** (Mean Absolute Error) : **4394.45**
- **RMSE** (Root Mean Squared Error) : **4800.59**
- **Coverage entre les quantiles 5% et 95%** : **79.17%**

### Interprétation rapide :
- **MAE et RMSE** sont relativement proches, ce qui indique des erreurs de prédiction cohérentes et pas dominées par quelques très gros écarts (peu d'outliers majeurs).
- **Coverage de 79%** signifie que **79% des vraies valeurs** sont bien tombées **entre les prédictions du quantile 5% et 95%**.  
  Cela montre que l’intervalle de prédiction est **un peu étroit**, car idéalement, pour un intervalle 5%–95%, on voudrait un **coverage proche de 90% ou un peu plus**.  
  → Ici, 79% indique **un modèle qui a encore du mal à bien estimer l'incertitude**.

**Remarque sur le Coverage** :  
- **Un intervalle 5%-95%** **ne** veut pas dire **95% de coverage garanti** !  
- En théorie, c’est sur des données "parfaitement modélisées" qu'on s'approcherait de **90%-95%** de couverture naturelle.  
- Si le modèle sous-estime la variance des séries temporelles, alors le coverage mesuré est plus faible, comme ici (~79%).
## Note – Résolution du problème d’environnement pour l’entraînement

Depuis **hier matin**, j'ai rencontré une série de **problèmes techniques** qui ont ralenti considérablement l'avancement.  
Voici le cheminement précis de ce qui s'est passé :

---

### 1. Origine du problème
- Au départ, une **erreur d'import** simple est survenue :  
  ➔ **Trainer** n'était **pas importé correctement** (j'avais utilisé `pytorch_lightning` au lieu de `lightning.pytorch`), ce qui a entraîné une cascade d'erreurs.
  
- Sur le moment, **pensant que le problème était plus grave**, j'ai cru qu'il fallait **tout recréer dans un environnement Docker** spécifique avec :
  - **Python 3.10**, 
  - des versions précises de **PyTorch, PyTorch Lightning, PyTorch Forecasting**, 
  - et une gestion GPU/CUDA propre.

---

### 2. Tentatives pour corriger
- **Création d'un Dockerfile** personnalisé,
- **Recherche des bonnes versions** de toutes les librairies,
- **Tentatives d'installation de CUDA Runtime**, 
- **Construction et rebuilds multiples** du Dockerfile,
- **Tests de détection GPU dans Docker**,  
- **Erreurs CUDA "unknown device"**, entraînant plusieurs tentatives de correction.

---

### 3. Problème aggravant
- À un moment donné, un **problème de Docker + GPU** est survenu.
- En réalité, **le problème principal** venait du fait que :
  - **Le GPU n'était plus reconnu correctement** par le système après une mise en veille du PC,
  - **Un simple redémarrage complet** du PC **aurait probablement suffi** à corriger l'environnement.
- Cette solution simple n'a pas été tentée tout de suite, par réflexe d'aller chercher des corrections complexes.

---

### 4. Solution trouvée
- Finalement, après avoir **abandonné Docker** pour ce projet :
  - **Création d'un environnement virtuel Python** (venv propre),
  - **Installation unique et simple** de :
    - `pip install pytorch_forecasting`
    - `pip install pandas matplotlib jupyterlab ipywidgets`
- Cela a suffi pour que **toutes les dépendances** (y compris `torch`, `lightning`, etc.) soient **automatiquement installées dans des versions compatibles**.

- **Vérification GPU** avec `torch.cuda.is_available()` :
  - Résultat : **TRUE**, **device disponible**, **entraînement fonctionnel**.
- **Entraînement réussi sans erreur**, **sans Docker**, **sans manipulation lourde**.

---

### 5. Conclusion pratique
- **Il n'était pas nécessaire** de créer un environnement Docker complexe pour ce projet.
- **Un environnement Python simple** suffit largement, à condition de :
  - **Installer uniquement `pytorch_forecasting`** (il gère ses dépendances internes),
  - **Vérifier CUDA avec `torch.cuda.is_available()`** avant l'entraînement.
- **Penser à redémarrer la machine** en cas de problème GPU inconnu avant de chercher des solutions lourdes.

---

**Bilan :**  
➔ **Pas besoin de Docker**,  
➔ **Pas besoin de forcer les versions**,  
➔ **Simple pip install + bon redémarrage + GPU détecté = Environnement parfaitement fonctionnel.**

## **Reconduction des entraînements (env. simple : i.e. environnement Python + pip install PyTorch Forecasting)**
Après reprise de l'entraînement dans un environnement Python classique (hors Docker), avec uniquement un `pip install pytorch_forecasting`, les résultats obtenus après 10 époques montrent des évolutions notables par rapport aux essais précédents :

- **Train Loss Step final** : **1060**
- **Val Loss final** : **855**
- **Train Loss Epoch final** : **1150**

**Paramètres d'entraînement** :
- Batch Size : **64**
- Max Encoder Length : **48**
- Max Prediction Length : **48**
- Hidden Size : **16**
- Hidden Continuous Size : **8**
- Output Size : **7** (par défaut pour quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
- Loss utilisée : **QuantileLoss** (non modifiée par rapport aux réglages initiaux)

**Résultats sur les prédictions** :
- MAE : **1760**
- RMSE : **2103**
- Coverage (entre 5% et 95% des quantiles) : **79%**

**Observations** :
- Contrairement à l'entraînement précédent, ici, **le Val Loss est resté inférieur au Train Loss**, ce qui est généralement bon signe.
- Aucun `seed` explicite n'a été fixé, ce qui peut expliquer de légères variations entre différents essais.
- Le changement de méthode d'environnement (passage hors Docker) a manifestement eu un **impact positif** sur la stabilité de l'entraînement.
### Pourquoi les scores sont meilleurs cette fois-ci ?

1. **Les données de validation ne sont pas les mêmes** qu’hier :
   - Hier, tu validais sur **2022** (*année complète, données jamais vues*).
   - Aujourd'hui, tu valides sur **2021**, c'est-à-dire **la dernière année de ton set d'entraînement**.
   - **Donc** : ton modèle a été entraîné sur **2015–2020** *et* validé sur **2021**, qui est **encore dans la distribution** du training. Cela rend **la validation beaucoup plus facile** pour le modèle.

2. **Pas de saut brutal dans les patterns temporels** :
   - De 2015 à 2021, ta série temporelle semble assez **homogène**.
   - Peut-être qu’en 2022 (l’année que tu utilisais hier pour tester), il y a eu **un changement de comportement** (ex : Covid, crises énergétiques, météo différente...).
   - Cela **explique pourquoi** ton modèle avait plus de mal à généraliser hier.

3. **Même modèle, même paramètres, mais data plus "facile"** :
   - Hidden Size, Encoder Length, Batch Size... tout est **quasiment pareil** entre hier et aujourd'hui.
   - Donc **l'amélioration** vient **presque exclusivement** du changement de la **zone temporelle** utilisée pour la validation.

4. **Pas lié à Docker / environnement / GPU** :
   - Le problème initial de performances n’était **pas lié au hardware**, ni à ton environnement logiciel.
   - C'était **purement une question de quelles données étaient utilisées pour la validation**.

---

### En résumé rapide :
- Hier tu validais sur **une année future (2022)** → données plus dures → scores mauvais (MAE ~ 2000, RMSE ~ 4000).
- Aujourd'hui tu valides sur **la dernière année de l'ensemble d'entraînement (2021)** → données familières → scores bien meilleurs (MAE ~ 1760, RMSE ~ 2100).
## 📋 Correction d'interprétation sur les résultats initiaux

Après analyse approfondie, il est confirmé que **les données d'entrée ont été normalisées** automatiquement lors de l'entraînement initial du modèle (*via le `GroupNormalizer` utilisé par défaut dans `pytorch_forecasting`*).

**Conséquences immédiates :**
- Les valeurs manipulées par le modèle étaient **de l'ordre de 0 à 10** (et non pas de 10 000 à 50 000 comme dans les données réelles de consommation).
- Les scores de **train loss**, **validation loss**, **MAE**, **RMSE** obtenus étaient donc **exprimés dans cet espace normalisé**, et **non sur les vraies valeurs**.
- Cela signifie que **l'interprétation directe des métriques** était **incorrecte** : les RMSE "de 4000" observés n'étaient **pas cohérents** avec les vraies échelles de consommation.

**Conclusion** :  
Tous les résultats produits avec la configuration précédente doivent être considérés **comme relatifs à un espace normalisé**. Ils **ne peuvent pas** être directement utilisés pour évaluer la qualité absolue du modèle sur les consommations en mégawattheures réels.

## Analyse résulats aprés running sans normalisation des consommation

## 1. **Normalisation**
**Pas de normalisation** ici.  
→ Tu as bien passé `target_normalizer=None` dans ton `TimeSeriesDataSet`.  
→ **Donc toutes les valeurs manipulées par le modèle sont sur l’échelle brute** (en MW), exactement comme dans ton CSV.

## 2. **Architecture du modèle**
- `hidden_size = 512`
- `attention_head_size = 16`
- `hidden_continuous_size = 124`
- `dropout = 0.1`
- `learning_rate = 0.01`

**C'est un gros modèle !**  
(Plus lourd que celui que tu avais lancé hier — normal vu les tailles que tu as mises.)

→ **Il est beaucoup plus capable d'apprendre**, mais il est aussi **beaucoup plus dur à entraîner** correctement.  
→ Ça justifie ton besoin d'avoir monté le learning rate (`0.01`).

## 3. **Observations sur les résultats**

Tu m’as dit :

|  | Résultat |
|:---|:---|
| **Train Loss** | ~3000 |
| **Val Loss** | ~4280 |
| **RMSE (Validation)** | ~9315 |
| **MAE (Validation)** | ~8648 |
| **Coverage (5%-95%)** | 79% |
| **Intervalle de confiance typique** | [33k, 96k] MW (≈ 30k MW d'écart) |

---

# Interprétation technique

### ➔ **Train/Val Loss**  
Le fait que ton **train_loss** soit autour de 3000 **et** que ton **val_loss** soit autour de 4300 est **cohérent** :

- Il y a **un petit écart raisonnable** → pas de surapprentissage brutal.
- Ça montre que ton modèle **arrive à apprendre une structure**, même si ce n'est pas parfait.

**MAIS** attention :  
→ La "Loss" ici est basée sur **le QuantileLoss**, pas un MAE brut, donc **ça donne une sorte d’erreur pondérée**, pas une erreur simple absolue.

---

### ➔ **RMSE et MAE**
- **MAE = ~8600**, **RMSE = ~9300** pour des consommations réelles qui varient entre 30k et 96k.
- **Erreur relative** (grossièrement) :  
\[
\text{Erreur relative} \approx \frac{9000}{65000} \approx 13-14\%
\]

Donc **ton modèle se trompe en moyenne de ~13-14%** sur la consommation →  
**Ce n’est pas excellent, mais ce n’est pas non plus catastrophique.**  
(En énergétique, surtout à horizon 48 pas, un RMSE de 10–15% est assez courant sans features météo.)

---

### ➔ **Intervalle de confiance**
- Ton intervalle de prédiction [33k, 96k] est **très large** (30k de spread).
- **Pourquoi ?**
  - Soit ton modèle a **encore une grosse incertitude** sur la dynamique future.
  - Soit l'apprentissage du quantile (QuantileLoss) est mal calibré car l'architecture est encore "instable".

C'est typique quand tu n'as pas encore bien **réglé les hyperparamètres** ou **filtré des features bruitées**.

---

# Résumé de ta situation

| Critère | Statut |
|:---|:---|
| Normalisation | Non (brut MW ) |
| Coût entraînement | Raisonnable vu la taille |
| Performance brute | Acceptable mais améliorable |
| Problèmes | Intervalle de confiance trop large (calibration mauvaise) |

---

# Conclusion rapide

- **Ton entraînement brut fonctionne bien.**
- **Ton modèle n'est pas "nul", mais pas encore ultra fiable**.
- **Prochaine étape** = Faire du tuning, réduire l'incertitude, jouer sur :
  - Learning rate
  - Taille modèle
  - Ajout de features (ex. jours fériés, température si dispo)
  - Augmentation du nombre d’époques pour stabiliser
