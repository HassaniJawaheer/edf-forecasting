# Journal des expérimentations

Entrées, sorties, résultats et limites des approches essayées. Les chiffres viennent des sorties enregistrées.

Les consommations concernent la France, pas uniquement les particuliers. Les erreurs sont exprimées en MW. MAE = erreur absolue moyenne ; RMSE = erreur qui donne davantage de poids aux gros écarts.

## 1. XGBoost — une journée complète en sortie

Code : [Eco2mixPreprocessGBoostDay](../src/edf_forecasting/components/eco2mix_preprocess_gboost_day.py), [Eco2mixTrainGBoostDay](../src/edf_forecasting/components/eco2mix_train_gboost_day.py). Pas de pipeline Kedro correspondant retrouvé dans le registre actuel.

- **Entrée :** sept journées précédentes par défaut, soit 336 consommations, auxquelles peuvent s’ajouter des variables des journées passées et du jour cible selon la configuration.
- **Sortie :** les 48 consommations de la journée suivante.
- **Modèle :** `MultiOutputRegressor(XGBRegressor)`, soit un modèle par créneau de sortie.
- **Résultats :** code retrouvé, mais aucun score sauvegardé retrouvé pour cette approche.
- **Point fort :** prévoit directement une journée entière, sans réinjecter les prédictions entre les 48 créneaux.
- **Limites :** entraînement de 48 modèles ; validation croisée en KFold dans le code ; les variables du jour cible doivent être disponibles au moment de prévoir.

## 2. train_xgboost_tabular — variables du même instant

Pipelines : `tune_xgboost_tabular`, `train_xgboost_tabular`.

- **Entrée :** productions par filière, échanges électriques, taux de CO₂ et variables calendaires du même créneau. La consommation cible et la colonne `tempo` sont retirées à l’entraînement.
- **Sortie :** une consommation pour ce même créneau de 30 minutes.
- **Modèle :** XGBoost, sans fenêtre de consommations passées dans cette branche.
- **Résultats :** pipelines retrouvés, mais aucun score sauvegardé retrouvé pour cette approche.
- **Point fort :** permet d’examiner l’apport de variables explicatives.
- **Limite :** les productions et échanges réalisés du futur ne sont pas connus au départ. Une bonne estimation avec ces variables ne prouverait pas une bonne prévision utilisable à l’avance.

Source : [entraînement tabulaire](../src/edf_forecasting/components/eco2mix_train_xgboost_tabular_30min.py).

## 3. conformal_prediction — XGBoost avec MAPIE

Source : [conformal_prediction.ipynb](../notebooks/conformal_prediction.ipynb).

- **Entrée :** 336 consommations réelles précédentes, soit sept jours.
- **Sortie :** la consommation suivante et un intervalle annoncé à 95 %.
- **Modèle :** XGBoost avec MAPIE, méthode `plus`, `cv=5`.
- **Découpage déclaré :** entraînement avant 2021, calibration sur 2021, test sur 2022.
- **Résultats affichés sur le test :** MAE **398,10 MW**, RMSE **527,80 MW**, couverture **6,50 %**, largeur moyenne **2 401,79 MW**.
- **Point fort :** première exploration des intervalles et des résidus ; le notebook contient aussi des appels de visualisation récursive.
- **Limites :** la couverture affichée est à revalider. L’extraction des bornes conserve potentiellement une dimension supplémentaire, ce qui peut fausser leur comparaison avec les observations. MAPIE est ajusté sur 2021 avec `cv=5` : ce n’est pas simplement la calibration d’un modèle figé. Ces chiffres ne permettent pas de conclure à un échec de MAPIE.

## 4. conformal_predictions_manual — calibration manuelle et récursion

Source : [conformal_predictions_manual.ipynb](../notebooks/conformal_predictions_manual.ipynb).

- **Entrée :** 336 consommations passées.
- **Sortie à l’entraînement :** la consommation suivante.
- **Sortie en utilisation récursive :** 48 consommations prédites, chaque prédiction étant réinjectée dans la fenêtre.
- **Modèle :** XGBoost ; intervalle calculé avec les quantiles des erreurs sur 2021.
- **Découpage déclaré :** entraînement avant 2021, calibration sur 2021, test sur 2022.

| Évaluation sauvegardée | Résultat |
|---|---:|
| MAE à un pas sur 2022 | 353,39 MW |
| RMSE à un pas sur 2022 | 462,17 MW |
| Couverture de l’intervalle à un pas | 95,56 % |
| Largeur moyenne de l’intervalle | 1 910,03 MW |
| Moyenne des RMSE des séquences récursives de 48 points sur 2021 | 2 212,04 MW |
| Médiane des RMSE de ces séquences | 1 627,80 MW |
| Plus grande RMSE de ces séquences | 10 528,07 MW |

- **Point fort :** les essais manuels mesuraient déjà la dégradation en récursion, sur plusieurs séquences, avec analyse des erreurs et de leur tendance.
- **Limites :** les intervalles sont calibrés à un pas, pas sur les trajectoires récursives. La moyenne des RMSE par séquence n’est pas la RMSE globale de notre backtest. Les points de départ et le modèle diffèrent également.

## 5. conformal_predictions_xgboost — fenêtre de 48 points et tuning

Source : [conformal_predictions_xgboost.ipynb](../notebooks/conformal_predictions_xgboost.ipynb), sorties du tuning datées du 14 mai 2025.

- **Entrée :** les 48 dernières consommations réelles.
- **Sortie :** la consommation suivante, avec un intervalle calibré sur 2021.
- **Modèle :** XGBoost ; recherche Optuna de 50 essais, validation en trois blocs KFold.
- **Résultats à un pas sur 2022 :** MAE **351,53 MW**, RMSE **481,68 MW**, couverture **94,16 %**, largeur moyenne **1 902,43 MW**.
- **Autre essai :** cinq trajectoires récursives de 48 points affichées avec une graine fixée à 123 ; pas de score global associé à ces cinq graphiques.
- **Point fort :** tuning et examen visuel du fonctionnement récursif dans le même notebook.
- **Limites :** le tuning ne respecte pas une validation passé-vers-futur ; les scores principaux restent ceux de la prévision à un pas.

## 6. xgboost_time_series — modèle utilisé par l’API

Pipelines : `tune_xgboost_time_series`, `train_xgboost_time_series`, regroupés avec la préparation des données dans `xgboost_time_series`.

- **Entrée :** 48 consommations passées. Les variables calendaires préparées ne sont pas utilisées par ce modèle.
- **Sortie à l’entraînement :** la consommation suivante.
- **Sortie dans l’API :** autant de valeurs que demandé, produites récursivement.
- **Données :** entraînement 2015–2020, calibration 2021, test 2022.

| Version sauvegardée | MAE à un pas sur 2022 | RMSE à un pas sur 2022 |
|---|---:|---:|
| 8 avril 2026 | 348,77 MW | 478,65 MW |
| 9 avril 2026 | 322,14 MW | 447,77 MW |
| 26 avril 2026 | 322,26 MW | 447,76 MW |

Pour la version du 26 avril : couverture **93,76 %** pour un intervalle annoncé à 95 %, largeur moyenne **1 708,77 MW**.

- **Point fort :** entraînement, calibration et évaluation intégrés à Kedro et MLflow ; modèles et scores sauvegardés.
- **Limites :** l’évaluation reçoit de vraies observations à chaque pas, alors que l’API réinjecte ses prédictions. Le tuning utilise KFold. Les intervalles ne sont pas calibrés par horizon.

Les pertes quantiles affichées dans les essais manuels, le notebook XGBoost et ce pipeline utilisent des niveaux incohérents avec les bornes à 95 % ; elles ne sont pas retenues comme résultats fiables ici.

Sources : [entraînement](../src/edf_forecasting/components/eco2mix_train_xgboost_time_series.py), [scores locaux versionnés](../data/07_model_output/eco2mix/time_series/30min/xgboost/evaluation/test_scores.yml/). Les scores dans `data` ne sont pas suivis par Git.

## 7. backtest_xgboost_time_series — comparaison à 24 heures et sept jours

Source : [pipeline](../src/edf_forecasting/pipelines/backtest_xgboost_time_series/nodes.py). Modèle du 26 avril 2026 réutilisé sans réentraînement.

- **Entrée du modèle :** 48 observations réelles au départ de chaque trajectoire.
- **Sortie :** 48 ou 336 prévisions, soit 24 heures ou sept jours.
- **Évaluation :** départ à minuit, 365 trajectoires de 24 heures et 359 trajectoires de sept jours sur 2021. Les observations de fin 2020 fournissent l’historique des premiers départs. Aucune cible de 2022.

| Méthode | Ce qui entre et ce qui sort | MAE sur 24 h | MAE sur 7 jours |
|---|---|---:|---:|
| XGBoost récursif | 48 observations au départ ; prédictions réinjectées | 2 224,47 MW | 4 684,48 MW |
| XGBoost avec observations actualisées | 48 vraies observations à chaque pas ; une prédiction à la fois | 298,29 MW | 296,50 MW |
| Dernière valeur répétée | Dernière observation copiée à tous les créneaux ; aucun modèle | 4 529,45 MW | 5 510,57 MW |
| Profil de la veille | 48 observations précédentes copiées, puis répétées si nécessaire ; aucun modèle | 2 783,02 MW | 4 527,81 MW |
| Profil de la semaine précédente | Valeurs des mêmes créneaux sept jours auparavant ; aucun modèle | 4 000,34 MW | 3 881,17 MW |

- **Point fort :** le modèle récursif bat les trois références sur l’ensemble des 24 heures.
- **Limites :** sur sept jours, il fait moins bien que les profils recopiés. Biais moyen de **−706,71 MW** sur 24 heures et **−1 394,32 MW** sur sept jours. La variante actualisée reçoit de nouvelles observations : elle n’est pas une prévision de toute la trajectoire disponible au départ.
- **Lecture des graphiques :** « médiane », « p90 » et « pire » sélectionnent des trajectoires selon l’erreur du modèle récursif. Aucune médiane n’est appliquée aux prédictions. Les fenêtres de sept jours se chevauchent.

[Rapport complet](rapports/backtest/91315a0c8fe84d86aff5a76ad1538ac8/results.md) · [Mesures par horizon](rapports/backtest/91315a0c8fe84d86aff5a76ad1538ac8/by_horizon.csv) · [Essai technique limité à trois départs](rapports/backtest/e63660e937ca4739afe0d969b8e0a691/results.md).

## 8. TFT — Temporal Fusion Transformer

Code : [tft_pipeline.ipynb](../notebooks/tft_pipeline.ipynb), [test_pipeline_tft.ipynb](../notebooks/test_pipeline_tft.ipynb) et scripts `notebooks/eco2mix_*tft.py`. Paramètres : [tft.yml](../conf/experiments/tft.yml).

- **Entrée :** 288 consommations passées (six jours), météo et calendrier. Le premier [test_tft.ipynb](../notebooks/test_tft.ipynb) utilise 48 consommations.
- **Sortie :** les 48 consommations de la journée suivante, avec plusieurs quantiles.
- **Essais :** variations du dropout, du nombre d’époques et de l’arrêt anticipé, avec PyTorch Forecasting et Lightning.

| Essai | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Premier `test_tft.ipynb` | Non enregistré | Non enregistré | Non enregistré |
| Notes : dropout 0,25, 50 époques | 991,44 | 1 413,38 | 0,83 |
| Notes : dropout 0,30, arrêt anticipé | 1 983,72 | 2 463,33 | 0,49 |
| `tft_pipeline.ipynb` | 4 418,79 | 4 626,13 | −1,81 |
| `evaluate_tft.ipynb`, données 2021 | 4 060,93 | 4 474,32 | −0,33 |
| Essai modulaire `20250505_212911` — score relevé avant suppression des artefacts | 30 919,44 | 31 042,51 | −125,38 |

- **Point positif :** une journée complète prédite directement, avec des variables supplémentaires.
- **Limites :** résultats très variables. Protocoles différents : pas de comparaison directe avec XGBoost. La validation du notebook principal ne couvre qu’une fenêtre finale ; la disponibilité de la météo future reste à vérifier.

Résultats détaillés : [notes TFT](experiences/tft/notes_resultats.md), [évaluation par fenêtres](../notebooks/evaluate_tft.ipynb). Environnement : [notes d’installation](experiences/tft/notes_installation.md).

## 9. LSTM — Long Short-Term Memory

Code : [test_lstm.ipynb](../notebooks/test_lstm.ipynb).

- **Entrée :** sept journées de 96 consommations chacune, sans variables supplémentaires.
- **Sortie :** les 96 consommations de la journée suivante.
- **Modèle :** LSTM PyTorch, 128 unités cachées, une couche récurrente, 50 époques, lots de 64.
- **Découpage :** 80 % entraînement, 20 % test, sans mélange.
- **Résultats :** RMSE **52 046,99** dans l’échelle des données, R² **−24,2882**. MAE : **non enregistré**.
- **Point positif :** prédiction directe de la journée complète.
- **Limites :** très mauvaises performances sur cet essai. Notebook à corriger avant relance : fonctions auxiliaires manquantes et incohérence entre les valeurs renvoyées et attendues.
