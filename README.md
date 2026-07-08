# Pipeline MLOps pour la Cybersécurité — Système de Détection d'Intrusions (IDS)
 
Pipeline MLOps de bout en bout qui entraîne, déploie, surveille et suit un modèle de machine learning pour la détection d'intrusions réseau, réalisé dans le cadre du **Projet de fin de Module — Machine Learning (CI2, Cybersécurité, 2025-2026)**.
 
---
 
## Vue d'ensemble
 
Les systèmes de détection d'intrusions (IDS) traditionnels, basés sur des signatures statiques, montrent des limites face aux attaques zero-day et aux comportements anormaux évolués. Ce projet propose un **IDS basé sur le Machine Learning**, intégré dans un pipeline MLOps complet — couvrant l'ensemble du cycle de vie, depuis les données brutes jusqu'à une API de production surveillée.
 
**Fonctionnalités du pipeline :**
1. Prétraitement des données de trafic réseau brutes (NSL-KDD)
2. Entraînement et comparaison de modèles ML supervisés (Random Forest, XGBoost, SVM)
3. Traçabilité des expériences et gestion des versions de modèles avec **MLflow**
4. Prédictions en temps réel via une API REST **FastAPI**
5. Containerisation de tous les services avec **Docker** / **docker-compose**
6. Automatisation des tests et du déploiement avec **GitHub Actions (CI/CD)**
7. Surveillance des performances de l'API en temps réel avec **Prometheus** et **Grafana**
8. Détection de dérive des données entre les données d'entraînement et de production avec **Evidently AI**
---
 
## Architecture
 
```
                ┌─────────────────────┐
                │  Données réseau      │
                │  brutes (NSL-KDD)    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prétraitement des     │
                │ données (encodage,    │
                │ mise à l'échelle)     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Entraînement du      │
                │  modèle (RF, XGBoost, │
                │  SVM) + suivi MLflow  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Meilleur modèle      │
                │  sauvegardé            │
                │  (best_model.pkl)     │
                └──────────┬───────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │            API REST FastAPI       │
        │   /predict  /health  /model-info  │
        │   expose les métriques Prometheus │
        └───────┬───────────────┬──────────┘
                │               │
                ▼               ▼
      ┌──────────────┐   ┌──────────────────┐
      │  Prometheus   │──▶│      Grafana      │
      │ (scrute l'API)│   │ (tableaux de bord │
      │               │   │  en temps réel)   │
      └──────────────┘   └──────────────────┘
 
      ┌───────────────────────────────────┐
      │           Evidently AI              │
      │  Compare les données d'entraîne-    │
      │  ment et de test                    │
      │  → rapport de dérive (HTML)         │
      └───────────────────────────────────┘
 
        Tous les services sont orchestrés
           ensemble via docker-compose.yml
```
 
**Services :**
| Service | Rôle |
|---|---|
| `api` | Sert les prédictions d'intrusion via FastAPI |
| `mlflow` | Trace les expériences, métriques et versions de modèles |
| `prometheus` | Scrute les métriques en direct de l'API |
| `grafana` | Visualise les métriques sur des tableaux de bord en temps réel |
| `evidently` (script) | Génère un rapport de dérive des données entre les jeux d'entraînement et de test |
 
---
 
## Structure du Projet
 
```
mlops-ids-project/
│
├── data/
│   └── raw/
│       ├── KDDTrain_.txt
│       └── KDDTest_.txt
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── api/
│   └── main.py                  # Application FastAPI (/predict, /health, /model-info)
│
├── models/
│   ├── best_model.pkl            # Classifieur XGBoost entraîné
│   ├── scaler.pkl                 # StandardScaler ajusté (41 caractéristiques)
│   └── encoders.pkl               # LabelEncoders ajustés (protocol_type, service, flag)
│
├── tests/
│   └── test_api.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   └── evidently_report.py       # Analyse de dérive des données
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
 
---
 
## Jeu de Données
 
**NSL-KDD**, un jeu de données de référence standard pour la recherche en détection d'intrusions réseau.
 
- Source : https://github.com/jmnwong
- Fichiers : `KDDTrain_.txt` (125 973 lignes), `KDDTest_.txt` (22 544 lignes)
- Placer les deux fichiers dans `data/raw/`
---
 
## Prérequis
 
```txt
pandas
numpy
scikit-learn
xgboost
mlflow
fastapi
uvicorn
prometheus-fastapi-instrumentator
evidently
pytest
requests
joblib
```
 
Installation :
```bash
pip install -r requirements.txt
```
 
---
 
## Démarrage
 
```bash
# 1. Cloner le dépôt
git clone https://github.com/elymou/mlops-ids-project.git
cd mlops-ids-project
 
# 2. Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
 
# 3. Installer les dépendances
pip install -r requirements.txt
 
# 4. Prétraiter les données
python src/data_preprocessing.py
 
# 5. Entraîner les modèles
python src/train.py
 
# 6. Lancer tous les services
docker compose up --build
```
 
> Désactiver l'environnement virtuel à tout moment avec `deactivate`. Le dossier `venv/` doit être exclu du dépôt Git via `.gitignore`.
 
Une fois lancés :
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Documentation API (Swagger) | http://localhost:8000/docs |
| Interface MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
 
---
 
## Utilisation de l'API
 
L'endpoint `/predict` attend un unique champ `features` — un tableau de 41 valeurs numériques prétraitées (encodées et mises à l'échelle) :
 
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.12, 0.0, 0.87, ... , 0.03]}'
```
 
`/model-info` et `/health` permettent de vérifier le statut ; la documentation interactive du schéma se trouve sur `/docs`.
 
---
 
## Surveillance (Monitoring)
 
- **Prometheus** collecte le nombre de requêtes, la latence, et des compteurs personnalisés (`ids_normal_traffic_total`, `ids_attacks_detected_total`) depuis l'endpoint `/metrics` de l'API.
- **Grafana** affiche des tableaux de bord visualisant le nombre total de requêtes, le taux de requêtes, le trafic normal et les attaques détectées en temps réel.
- **Evidently AI** (`monitoring/evidently_report.py`) compare les jeux de données d'entraînement et de test et génère un rapport de dérive au format HTML :
```bash
cd monitoring
python3 evidently_report.py
```
 
Cela produit `evidently_report.html`, qui identifie les caractéristiques ayant dérivé entre la distribution de référence (entraînement) et la distribution actuelle (test), à l'aide du test de distance de Wasserstein.
 
---
 
## Portée et Limites
 
Ce projet met en œuvre une **API de prédiction et un cycle de vie MLOps** autour d'un modèle IDS entraîné — il n'effectue pas de capture de paquets réseau en temps réel. L'API renvoie une prédiction pour tout tableau `features` qu'elle reçoit ; la connexion à un outil de capture de paquets en temps réel (par exemple Zeek ou Suricata) permettant d'extraire automatiquement les caractéristiques du trafic en direct dépasse le cadre de ce projet, mais constituerait la suite logique pour un déploiement en production.