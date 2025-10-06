# Challenge IA - Tâche 1 : Reconnaissance Faciale Robuste

Ce dépôt contient notre solution complète pour la Tâche 1 du challenge d'Intelligence Artificielle, axée sur la reconnaissance faciale robuste. L'objectif est d'apparier 2000 images de test (formant 1000 paires uniques) de personnes prises à différents moments de leur vie.

Notre approche combine une analyse rigoureuse des données, une stratégie de fine-tuning de modèle State-Of-The-Art, et un pipeline d'inférence robuste pour garantir la meilleure performance possible.

## Notre Philosophie

Face à un problème de reconnaissance faciale "in the wild" (dans des conditions non contrôlées), nous avons adopté une philosophie en trois points :
1.  **La Qualité des Données d'Abord :** Un bon modèle ne peut être construit que sur de bonnes données. Notre première étape a été de valider et de préparer un dataset d'entraînement propre et fiable.
2.  **S'appuyer sur l'État de l'Art :** Plutôt que de réinventer la roue, nous avons choisi de fine-tuner un modèle de reconnaissance faciale de classe mondiale (`FaceNet - InceptionResnetV1`) qui a déjà été pré-entraîné sur des millions de visages.
3.  **Apprentissage Ciblé et Intelligent :** Nous avons spécialisé ce modèle expert pour notre dataset spécifique en utilisant une technique de **Online Hard Triplet Mining**, le forçant à se concentrer sur les cas de reconnaissance les plus difficiles et ambigus.

## Pipeline de la Solution

Notre solution est structurée en trois phases claires, orchestrées dans un notebook Kaggle unique :

### 1. Préparation du Dataset
Conscients que les données initialement fournies pouvaient contenir des erreurs, nous avons utilisé le dataset de référence **CACD-VS** comme notre "Gold Standard". Pour simuler fidèlement les conditions du challenge, nous avons :
-   **Partitionné** le dataset de 4000 paires en un set d'entraînement (3000 paires) et un set de test interne (1000 paires).
-   **Anonymisé** notre set de test interne pour valider notre algorithme de matching dans des conditions réelles.

### 2. Fine-Tuning du Modèle
C'est le cœur de notre solution. Nous avons fine-tuné un modèle **`InceptionResnetV1`** (pré-entraîné sur VGGFace2) en utilisant une **perte Triplet (`TripletMarginLoss`)** avec **Online Hard Mining**.
-   **Pourquoi ce choix ?** Cette technique est extrêmement efficace. Au lieu d'apprendre sur des exemples faciles, le modèle se concentre à chaque étape sur les "triplets" les plus difficiles :
    -   L'**ancre** (une image).
    -   Le **positif le plus difficile** (l'autre image de la même personne qui lui ressemble le moins).
    -   Le **négatif le plus difficile** (l'image d'une autre personne qui lui ressemble le plus).
-   Ce processus force le modèle à apprendre un espace d'embedding extrêmement discriminant, robuste aux variations d'âge, de pose et d'éclairage.

### 3. Inférence et Matching
Une fois notre modèle fine-tuné, nous l'utilisons pour l'inférence sur le set de test :
-   **Génération d'Embeddings :** Chaque image de test est transformée en un vecteur de caractéristiques (embedding) de 512 dimensions.
-   **Matching Robuste :** Nous calculons une matrice de similarité cosinus entre tous les embeddings. Ensuite, un algorithme de **matching glouton** est utilisé pour trouver les 1000 paires les plus probables en s'assurant qu'aucune image n'est utilisée plus d'une fois.

## Comment Exécuter le Projet

Ce projet est conçu pour être exécuté dans un environnement de notebook Kaggle avec un accélérateur GPU.

1.  **Prérequis :**
    -   Créez un dataset Kaggle contenant le dataset **CACD-VS** et ajoutez-le à votre notebook.
    -   Assurez-vous que l'option "Internet" est activée dans les paramètres du notebook.

2.  **Structure du Notebook :**
    Le notebook est divisé en cellules logiques :
    -   **Cellule 1 :** Installe les dépendances (`facenet-pytorch`, etc.) et importe les bibliothèques.
    -   **Cellule 2 :** Prépare les données en créant les dossiers `train` et `test` à partir du dataset source.
    -   **Cellule 3 :** Définit les classes `Dataset` et `collate_fn`.
    -   **Cellule 4 :** Exécute la boucle de fine-tuning du modèle.
    -   **Cellule 5 :** Exécute l'inférence sur le set de test, génère le fichier `submission.csv` et affiche une visualisation des 10 meilleures paires trouvées.

3.  **Exécution :**
    -   Vérifiez le chemin vers le dataset source dans la `Cellule 2`.
    -   Exécutez les cellules séquentiellement de haut en bas.

## Résultats
L'entraînement de notre modèle a montré une convergence stable et une perte finale très basse, indiquant un apprentissage réussi. La validation visuelle des paires générées sur le set de test a confirmé la haute précision et la robustesse de notre modèle face à des variations significatives d'apparence.

Le fichier `submission.csv` produit contient les 1000 paires d'images identifiées par notre pipeline.