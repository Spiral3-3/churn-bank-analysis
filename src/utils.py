import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from itertools import cycle

def find_discretization_thresholds(x, y, nb_bins_max, eff_min):
    """
    Identifie les seuils optimaux pour discrétiser une variable continue x
    afin d'expliquer au mieux une variable binaire y en utilisant un arbre de décision.
    
    Parameters:
    - x: np.array, variable continue à discrétiser
    - y: np.array, variable binaire à expliquer
    - nb_bins_max: int, nombre maximum de classes (bins)
    - eff_min: int, effectif minimum toléré par classe
    
    Returns:
    - list: liste des seuils triés dans l'ordre croissant
    """
    # Reshape x pour sklearn (doit être 2D)
    x_reshaped = x.reshape(-1, 1)
    
    # Créer et entraîner l'arbre de décision
    dt = DecisionTreeClassifier(
        criterion='gini',  # Ou 'entropy' si préféré
        max_leaf_nodes=nb_bins_max,
        min_samples_leaf=eff_min,
        random_state=42  # Pour reproductibilité
    )
    dt.fit(x_reshaped, y)
    
    # Accéder à la structure interne de l'arbre
    tree = dt.tree_
    
    # Collecter les seuils (thresholds) pour la feature 0 (seule feature)
    thresholds = []
    for i in range(tree.node_count):
        if tree.feature[i] == 0:  # Seulement pour notre feature
            thresholds.append(tree.threshold[i])
    
    # Supprimer les valeurs invalides (comme -2.0 qui indique une feuille)
    thresholds = [t for t in thresholds if t != -2.0]
    
    # Supprimer les doublons et trier
    thresholds = sorted(set(thresholds))
    
    return thresholds

# Fonction d'application des seuils de mise en classes
def assign_class(value, thresholds):
    if len(thresholds) == 0:
        return f"< {thresholds[0]}"
    for i in range(len(thresholds)):
        if value < thresholds[0]:
            return f"0. <= {thresholds[0]}"
        if value >= thresholds[-1]:
            return f"{len(thresholds)}. > {thresholds[-1]}"
        if thresholds[i] <= value < thresholds[i + 1]:
            return f"{i+1}. ]{thresholds[i]}, {thresholds[i+1]}]"

def roc_curve_compare(dic_roc, title):
    colors = cycle(['seagreen', 'red', 'darkorange', 'yellowgreen', 'blue', 'gold'])
    fig, ax = plt.subplots(figsize=(7,6), dpi = 120)
    ax.set_aspect('equal', adjustable='box')
    for mod, color in zip(dic_roc.keys(), colors):
        fpr = dic_roc[mod]['fpr']
        tpr = dic_roc[mod]['tpr']
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, color=color, label=mod + ' (AUC = %0.3f)' % roc_auc)

    roc_curve_design(title)

# Fonction permettant de mettre en forme le graphique de comparaison des courbes Roc
def roc_curve_design(title):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', label='Sélection aléatoire', alpha=.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid(True, which='major', axis='x', alpha=0.4)
    plt.grid(True, which='major', axis='y', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize = 8)
    plt.show()

# Fonction permettant d'éditer la matrice de confusion, le rapport et la courbe ROC pour les échantillons d'apprentissage et test d'un modèle
# Nécessite d'avoir calculé au préalable y_pred_train, y_pred_test, proba_train et proba_test
def mat_conf_roc_curve(dic_roc, mod, y_train, y_test, y_pred_train, y_pred_test, proba_train, proba_test):
    fig, ax = plt.subplots(figsize=(7,6), dpi = 200)
    ax.set_aspect('equal', adjustable='box')
    for y_obs, y_pred, proba, ech_name, c in zip([y_train, y_test], [y_pred_train, y_pred_test], [proba_train, proba_test], ["train", "test"], ['blue', 'red']):
        print('Echantillon ' + ech_name)
        print(confusion_matrix(y_obs, y_pred))
        print(classification_report(y_obs, y_pred))    
        
        fpr, tpr, thresholds = roc_curve(y_obs, proba)
        # Sauvegarde des éléments de l'échantillon test pour comparaison avec les autres modèles
        if ech_name == 'test' :
            dic_roc[mod] = {}
            dic_roc[mod]['fpr']= fpr
            dic_roc[mod]['tpr']= tpr
        roc_auc = auc(fpr, tpr)
        print('Modèle {} - Echantillon {} - Aire sous la courbe ROC (AUC) : {:0.1%}'.format(mod, ech_name, roc_auc))
        
        plt.plot(fpr, tpr, lw=1, color = c,
                 label='Echantillon {} ({:0.1%})'.format(ech_name, roc_auc))
        
    roc_curve_design(mod + ' - Courbe ROC')

def mat_conf(dic_roc, mod, y_train, y_test, y_pred_train, y_pred_test, proba_train, proba_test):
    for y_obs, y_pred, proba, ech_name, c in zip([y_train, y_test], [y_pred_train, y_pred_test], [proba_train, proba_test], ["train", "test"], ['blue', 'red']):
        print('Echantillon ' + ech_name)
        print(confusion_matrix(y_obs, y_pred))
        print(classification_report(y_obs, y_pred))    
        
        fpr, tpr, thresholds = roc_curve(y_obs, proba)
        # Sauvegarde des éléments de l'échantillon test pour comparaison avec les autres modèles
        if ech_name == 'test' :
            dic_roc[mod] = {}
            dic_roc[mod]['fpr']= fpr
            dic_roc[mod]['tpr']= tpr
        roc_auc = auc(fpr, tpr)

def analyze_target_rate_by_decile(mod, num_quantiles,  y_train, y_test, proba_train, proba_test):
    """
    Analyse le taux de cible (y=1) par décile de score pour des échantillons d'apprentissage et de test.

    Parameters:
        proba_train (array-like): Scores de l'échantillon d'apprentissage.
        proba_test (array-like): Scores de l'échantillon test.
        y_train (array-like): Valeurs cibles de l'échantillon d'apprentissage (0 ou 1).
        y_test (array-like): Valeurs cibles de l'échantillon test (0 ou 1).
        num_quantiles (int): Nombre de quantiles (doit être > 1).

    Returns:
        None
    """
    # Validation des entrées
    if num_quantiles <= 1:
        raise ValueError("Le nombre de quantiles doit être supérieur à 1.")

    # Combiner les probabilités pour déterminer les seuils des déciles
    quantile_bins0 = np.quantile(combined_proba := np.concatenate([proba_train, proba_test]),
                                 q=np.linspace(0, 1, num_quantiles+1))
    quantile_bins = np.unique(quantile_bins0)

    if len(quantile_bins) - 1 < num_quantiles:
        print(f"Attention : certains bins sont dupliqués. Ajustement à {len(quantile_bins) - 1} bins.")

    # Découpage des scores en fonction des quantiles
    train_bins = pd.cut(proba_train, bins=quantile_bins, labels=False, include_lowest=True)
    test_bins = pd.cut(proba_test, bins=quantile_bins, labels=False, include_lowest=True)

    # Inverser les numéros de quantiles pour que 1 soit associé aux plus petites valeurs
    if len(quantile_bins) > 2:
        train_bins = (num_quantiles - 1) - train_bins
        test_bins = (num_quantiles - 1) - test_bins

    # Préparer les DataFrames
    train_df = pd.DataFrame({"score_bin": train_bins, "target": y_train})
    test_df = pd.DataFrame({"score_bin": test_bins, "target": y_test})

    # Calculer le taux de cible par décile
    train_results = train_df.groupby("score_bin")["target"].mean().reset_index()
    train_results.rename(columns={"target": "target_rate_train"}, inplace=True)

    test_results = test_df.groupby("score_bin")["target"].mean().reset_index()
    test_results.rename(columns={"target": "target_rate_test"}, inplace=True)

    # Fusionner les résultats pour un affichage dans un tableau
    results = pd.merge(train_results, test_results, on="score_bin", how="outer").fillna(0)
    results["score_bin"] = results["score_bin"].astype(int) + 1
    print([0]+list(quantile_bins))

    # Ajuster les libellés pour regrouper les bins dupliqués selon quantile_bins0

    bin_labels = [1]
    quantile_bins0 = -np.sort(-quantile_bins0)
    previous_bin = quantile_bins0[0]
    range_start = 1
    for i in range(1, len(quantile_bins0)):
        if quantile_bins0[i] == previous_bin:
            bin_labels[-1] = f"{range_start}-{i+1}"
        else:
            bin_labels.append(str(i+1))
            range_start = i+1
        previous_bin = quantile_bins0[i]

    results["bin_label"] = bin_labels[:len(results)]

    # Affichage du tableau
    print(results)

    # Création du graphique
    plt.figure(figsize=(10, 6))
    bar_width = 0.4

    x_positions = np.arange(len(results))

    train_bars = plt.bar(x_positions - bar_width / 2, results["target_rate_train"],
                         color="navy", alpha=0.7, width=bar_width, label="Taux de cible - Train")
    test_bars = plt.bar(x_positions + bar_width / 2, results["target_rate_test"],
                        color="crimson", alpha=0.7, width=bar_width, label="Taux de cible - Test")

    plt.title("Modèle " + mod + " - Taux de cible par décile de score")
    plt.xlabel("Décile de score")
    plt.ylabel("Taux de cible (y=1)")
    plt.xticks(x_positions, results["bin_label"], rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les annotations
    for bar in train_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    for bar in test_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()

def RF_feature_importance(rf, feature_names):
    # Récupération des importances avec les noms des colonnes
    importances = rf.feature_importances_

    # Création d'un DataFrame trié
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Affichage sous forme de tableau
    print(feature_importance_df)

    # Optionnel : graphique horizontal
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Importance des variables - Random Forest')
    plt.gca().invert_yaxis()  # Pour avoir la plus importante en haut
    plt.show()

def plot_metrics_vs_threshold(y_true, y_proba, n_thresholds=100, return_data=False):
    """
    Trace Precision, Recall, FPR et Accuracy en fonction du seuil de probabilité.
    
    Paramètres :
    - y_true : array-like, vraies étiquettes (0/1)
    - y_proba : array-like, probabilités prédites pour la classe positive
    - n_thresholds : int, nombre de seuils à évaluer (plus = plus lisse)
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # Grille de seuils de 0 à 1 (on évite exactement 0 et 1 pour éviter des divisions par zéro)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    precisions = []
    recalls = []
    fprs = []
    accuracies = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Precision
        if np.sum(y_pred == 1) > 0:
            precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
        else:
            precision = 1.0  # Aucun positif prédit → precision = 1 par convention
        precisions.append(precision)
        
        # Recall (TPR)
        if np.sum(y_true == 1) > 0:
            recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
        else:
            recall = 0.0
        recalls.append(recall)
        
        # FPR
        neg_mask = (y_true == 0)
        if np.sum(neg_mask) > 0:
            fpr = np.sum((y_pred == 1) & neg_mask) / np.sum(neg_mask)
        else:
            fpr = 0.0
        fprs.append(fpr)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_true)
        accuracies.append(accuracy)
    
    # Tracé
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, label='Precision', color='blue', lw=2)
    plt.plot(thresholds, recalls, label='Recall (TPR)', color='green', lw=2)
    plt.plot(thresholds, fprs, label='FPR', color='red', lw=2)
    plt.plot(thresholds, accuracies, label='Accuracy', color='purple', lw=2)
    
    plt.xlabel('Seuil de probabilité (Threshold)', fontsize=14)
    plt.ylabel('Valeur de la métrique', fontsize=14)
    plt.title('Scoring - Aide au choix du seuil', fontsize=16)
    plt.legend(loc='center right', fontsize=12)
    plt.grid(True, alpha=0.3)
    # plt.gca().invert_xaxis()  # Seuil élevé à gauche, bas à droite

    # Axe X : pas de 0.05
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    
    # Axe Y : en pourcentage, pas de 5%
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Ligne verticale au seuil classique 0.5
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
    plt.text(0.52, 0.95, 'Seuil = 0.5', rotation=90, verticalalignment='top', 
             color='gray', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Optionnel : retourner les valeurs pour analyse ultérieure
    if return_data:
        return {
            'thresholds': thresholds,
            'precision': np.array(precisions),
            'recall': np.array(recalls),
            'fpr': np.array(fprs),
            'accuracy': np.array(accuracies)
        }
    return None


def analyze_score_deciles(mod, num_quantiles, y_train, y_test, proba_train, proba_test):
    if num_quantiles <= 1:
        raise ValueError("num_quantiles doit être > 1")

    combined_proba = np.concatenate([proba_train, proba_test])
    bins = np.unique(np.quantile(combined_proba, np.linspace(0, 1, num_quantiles + 1)))

    train_bins = pd.cut(proba_train, bins=bins, labels=False, include_lowest=True)
    test_bins  = pd.cut(proba_test,  bins=bins, labels=False, include_lowest=True)

    train_bins = (len(bins) - 2) - train_bins
    test_bins  = (len(bins) - 2) - test_bins

    train_df = pd.DataFrame({"decile": train_bins + 1, "target": y_train})
    test_df  = pd.DataFrame({"decile": test_bins + 1,  "target": y_test})

    train_rate = train_df.groupby("decile")["target"].mean()
    test_rate  = test_df.groupby("decile")["target"].mean()

    results = pd.concat([train_rate.rename("Train"), test_rate.rename("Test")], axis=1).fillna(0)

    print(f"\nAnalyse par décile – {mod}")
    print(results)

    ax = results.plot(kind="bar", figsize=(9, 5), alpha=0.8, grid=True)
    plt.title(f"{mod} – Taux de cible par décile")
    plt.xlabel("Décile (1 = risque élevé)")
    plt.ylabel("Taux de cible")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return results
