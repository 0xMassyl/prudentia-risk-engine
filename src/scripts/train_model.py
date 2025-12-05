import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from src.processing.pipeline import create_scoring_pipeline

# Configuration
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "scorecard_model.pkl"
DATA_SIZE = 5000

def generate_dummy_data(n_samples: int = 1000):
    """
    Génère un dataset financier synthétique crédible pour tester le pipeline.
    """
    np.random.seed(42)
    
    data = pd.DataFrame({
        # Variables Numériques
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'loan_amount': np.random.normal(150000, 50000, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        
        # Variables Catégorielles
        'sector': np.random.choice(['Tech', 'Construction', 'Finance', 'Retail', 'Energy'], n_samples),
        'rating_agency': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'C'], n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.15]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })
    
    # Création d'une target corrélée (pour que le modèle apprenne quelque chose)
    # Les secteurs 'Construction' et les ratings bas ('C') augmentent le risque de défaut
    risk_score = (
        (data['income'] < 30000) * 2 +
        (data['rating_agency'] == 'C') * 3 +
        (data['sector'] == 'Construction') * 1.5 -
        (data['years_employed'] > 10) * 1
    )
    
    # Conversion en probabilité via Sigmoid + Bruit
    proba = 1 / (1 + np.exp(-(risk_score - 2)))
    data['default_flag'] = np.random.binomial(1, proba)
    
    return data

def main():
    print("Démarrage du training pipeline...")
    
    # 1. Génération des données
    print(f"Génération de {DATA_SIZE} lignes de données synthétiques...")
    df = generate_dummy_data(DATA_SIZE)
    
    X = df.drop(columns=['default_flag'])
    y = df['default_flag']
    
    print(f"Taux de défaut simulé : {y.mean():.2%}")
    
    # 2. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Définition des features
    numeric_features = ['age', 'income', 'loan_amount', 'years_employed']
    categorical_features = ['sector', 'rating_agency', 'region']
    
    # 4. Création et Entraînement du Pipeline
    print("Entraînement du modèle (WoE + Logistic Regression)...")
    pipeline = create_scoring_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)
    
    # 5. Évaluation
    print("Évaluation des performances...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    
    print("\n RESULTATS:")
    print(f"   ROC AUC : {auc:.4f}")
    print(f"   GINI    : {gini:.4f} (Standard Bancaire > 0.40 attendu)")
    
    print("\n   Rapport de classification (Seuil 0.5) :")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 6. Sauvegarde
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")
    print(" Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()