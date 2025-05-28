# Projet CNN avec Transfert Learning, MLOps et Déploiement API

## Description

Ce projet utilise un modèle CNN (VGG16) en transfert learning pour la classification d’images.  
Il intègre un pipeline MLOps simple pour :  
- gérer l’entraînement et le suivi des expériences avec MLflow,  
- sauvegarder et versionner les modèles,  
- déployer une API de prédiction avec FastAPI.

---

## Prérequis

- Python 3.6+  
- Virtualenv ou venv  
- Packages listés dans `requirements.txt`

---

## Installation

1. Cloner le dépôt :

```bash
git clone 'https://github.com/amina-abddm/Brief_CNN_Transfer_Learning.git'
cd 'Biref_CNN_Transfer_Learning'

2. Créer et activer un environnement virtuel : 

python3 -m venv .venv
source .venv/bin/activate


3. Installation des requirements : 

pip install -r Requirements.txt


## Structure du projet : 

Brief_CNN_Transfer_Learning/

├── .venv/                 # Environnement virtuel
├── data/chest_xray        # Données d'entrée
│   ├── test
│   ├── train     
│   └── val 
├── mlruns/                # Dossier MLflow (créé automatiquement)
├── models/                # Modèles sauvegardés   
│   ├── cnn_pneumonia.h5   # Sauvegarde du modéle d'entrainement CNN
│   ├── transfer_learning_pneumonia.keras                    # Sauvegarde du modéle de pré-entrainement VGG16
│   └── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5    # Poids du modèle VGG16 pré-entraîné sans les couches top 
├── .gitignore
├── README.md              # Documentation du projet 
└── requirements.txt       # Dépendances Python


