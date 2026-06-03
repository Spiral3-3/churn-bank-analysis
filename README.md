# 🏦 Projet Scoring : Prédiction du Churn Bancaire

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![Cadre](https://img.shields.io/badge/Cadre-BUT_3_SD_Vannes-success)

## Contexte du Projet
Ce dépôt contient le code et les résultats du projet sur la prédiction du churn. 

Dans le secteur bancaire, anticiper le départ des clients (Churn) est un enjeu stratégique majeur. Notre travail consiste à élaborer un modèle de machine learning permettant de prédire au mieux la clôture des comptes clients. L'approche se veut orientée "Business" : l'objectif final est d'évaluer l'apport financier et stratégique du modèle pour la banque en gérant efficacement le taux de faux positifs.

## Les Données
L'analyse repose sur un jeu de données de **12 300 clients**. Pour chaque profil, le modèle exploite les informations suivantes :
* **Démographie & Profil :** Âge, Genre, Zone géographique.
* **Finances :** Score de risque de crédit, Solde bancaire, Salaire estimé.
* **Comportement & Engagement :** Ancienneté, Nombre de produits détenus (épargne, assurances), Utilisation d'une carte de crédit, Activité sur le compte.
* **Variable cible :** `Churn` (Oui/Non).

## Méthodologie
Notre démarche analytique respecte les étapes suivantes :
1. **Data Préparation :** Nettoyage, recodages, filtres et création de nouvelles variables pertinentes.
2. **Analyse Exploratoire :** Mise en évidence du profil-type des clients "churners" (ceux qui quittent la banque).
3. **Modélisation :** Entraînement et comparaison de plusieurs algorithmes. Le projet intègre notamment **un arbre de décision**, **plusieurs régressions logistiques** et **une forêt aléatoire**.
4. **Évaluation Métier :** Sélection du meilleur modèle, choix des seuils de probabilité optimaux, et calcul des métriques clés (% de churners correctement identifiés, minimisation des faux positifs).

## Exécuter le code
L'ensemble des traitements et la modélisation sont centralisés dans un unique fichier Python. 
