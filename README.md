# Projet CNN avec Transfert Learning, MLOps et Déploiement API

## 🧠 Description du projet

Ce projet utilise un modèle CNN (VGG16) en transfert learning pour la classification d’images.  
Il intègre un pipeline MLOps simple pour :  

- gérer l’entraînement et le suivi des expériences avec MLflow,  
- sauvegarder et versionner les modèles,  
- déployer une API de prédiction avec FastAPI.

---

## 📦 Prérequis du prjot

- Python 3.12+  
- Virtualenv ou venv  
- Packages listés dans `requirements.txt`

---

## ⚙️ Installation du projet

1. ### 📥 Cloner le dépôt

''' bash
git clone 'https://github.com/amina-abddm/Brief_CNN_Transfer_Learning.git'
cd 'Biref_CNN_Transfer_Learning'

2. ### ✨ Créer et activer un environnement virtuel

'''bash
python3 -m venv .venv
source .venv/bin/activate

3. ### ⬇️ Installation des requirements

''' bash
pip install -r Requirements.txt

---

## 🧱 Structure du projet – CNN Transfer Learning Classification (MLOps)

Brief_CNN_Transfer_Learning/

├── .venv/
├── data/chest_xray
│   ├── test
│   ├── train
│   └── val
├── mlruns/
├── models/
│   ├── cnn_pneumonia.h5  
│   ├── transfer_learning_pneumonia.keras
│   └── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5    # Poids du modèle VGG16 pré-entraîné sans les couches top 
├── .gitignore
├── README.md
└── requirements.txt

```
Brief_CNN_Transfer_Learning/

├── .venv/
├── data/chest_xray
│   ├── test
│   ├── train
│   └── val
├── mlruns/
├── models/
│   ├── cnn_pneumonia.h5  
│   ├── transfer_learning_pneumonia.keras
│   └── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5    # Poids du modèle VGG16 pré-entraîné sans les couches top 
├── .gitignore
├── README.md
└── requirements.txt
```

### 📂 Dossier et fichiers – Explication

.venv/

- Contient l’environnement virtuel Python local (activé avec source .venv/bin/activate)
- Non versionné grâce au .gitignore
- Assure l’isolation des dépendances du projet

data/chest_xray/

- Données d’entrée classées selon les standards de test / train / val
- Utilisées pour l'entraînement, la validation, et l’évaluation du modèle
- Structure héritée d’un dataset d’imagerie médicale type Kaggle

mlruns/

- Répertoire généré automatiquement par MLflow
- Stocke tous les runs d'expérimentation : paramètres, métriques, artefacts
- Permet un suivi structuré et visualisable (mlflow ui) des essais

models/

- Contient les modèles entraînés et leurs poids
- Ce dossier facilite la version des modèles et la réutilisation directe en évaluation ou en production.

| Fichiers                                      | Descriptions                                             |
| ----------------------------------------------|---------------------------------------------------       |
| cnn\_pneumonia.h5                             | Modèle CNN entraîné à partir de zéro (baseline)          |
| transfer\_learning\_pneumonia.keras           | Modèle VGG16 avec transfert learning, adapté aux données |
| vgg16\_weights\_tf\_dim\_ordering\_tf\_kernels\_notop.h5 (include\_top=False) | Poids pré-entraînés de VGG16 sans les couches finales    |

.gitignore

- Exclut de Git les dossiers comme .venv/, __pycache__/, mlruns/, etc.
- Garantit un dépôt propre, sans fichiers lourds ou spécifiques à une machine

README.md

- Fichier principal de documentation
- Explique : objectif, installation, utilisation, organisation du projet

Requirements.txt

- Liste toutes les dépendances Python nécessaires au projet : tensorflow, mlflow, numpy, etc.
- Permet de recréer facilement l’environnement sur une autre machine

## 🎯 Ce que permet cette structure en MLOps

Cette structure a été pensée pour intégrer les bonnes pratiques de MLOps tout en restant simple à maintenir. Elle permet de structurer le cycle de vie du modèle machine learning de manière claire, traçable et reproductible.

| Objectif                              | Élément correspondant                                  |
|-------------------------------------  |  -------------------------------------------           |
| 🗂️ Organisation claire                | `data/`, `models/`, `.venv/`                            |
| ⚙️ Entraînement & réutilisation       | `cnn_pneumonia.h5`, `transfer_learning_pneumonia.keras` |
| 📊 Suivi d’expérimentations           | `mlruns/` (via **MLflow**)                              |
| ♻️ Réplication de l’environnement     | `.venv/`, `requirements.txt`                            |
| 📝 Documentation pour les utilisateurs| `README.md`                                             |

Cette organisation facilite également le passage à une structure plus avancée (déploiement, CI/CD, monitoring) en cas d’évolution vers un pipeline MLOps complet.

---

## 📋 Utilisation Mlflow

1. ### 📊 Suivi des expériences (MLflow)

'''
bash
mlflow ui --port 5001
'''
➡️ Pour lancer mlflow ui et suivre les runs, ouvrir dans un navigateur : http://127.0.0.1:5001

---

## ✅ Conclusion

Le projet montre que les modèles CNN, notamment avec le transfert learning (VGG16), détectent efficacement les cas de pneumonie à partir d’images médicales.  
Cependant, les résultats sont moins bons pour les cas normaux : cela indique qu’un **meilleur équilibre des données** ou un **ajustement de l'entraînement** est nécessaire pour améliorer la performance globale du modèle.

Grâce à une structure organisée et à l'utilisation de MLflow, ce projet pose une base solide pour évoluer vers un pipeline MLOps plus complet.

---

### 💡 Étapes suivantes possibles

- Ajout d’une API pour exposer le modèle (FastAPI)
- Mise en production du pipeline via Docker
- Intégration de tests et automatisation (CI/CD)
