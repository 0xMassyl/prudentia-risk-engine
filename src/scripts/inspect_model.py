import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "data/models/scorecard_model.pkl"

def inspect():
    print(f"🔍 Chargement du modèle depuis {MODEL_PATH}...")
    
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Accès au modèle de Régression Logistique (dernière étape du pipeline)
    log_reg = pipeline.named_steps['classifier']
    
    print("\n📊 --- INTERCEPT & COEFFICIENTS ---")
    print(f"Intercept (Biais de base) : {log_reg.intercept_[0]:.4f}")
    
    # Récupération des noms de features (un peu technique avec ColumnTransformer)
    # On sait que l'ordre est : Numériques puis Catégorielles
    numeric_features = ['age', 'income', 'loan_amount', 'years_employed']
    categorical_features = ['sector', 'rating_agency', 'region']
    feature_names = numeric_features + categorical_features
    
    coeffs = log_reg.coef_[0]
    
    df_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coeffs,
        'Impact': np.abs(coeffs) # Pour trier par importance
    }).sort_values('Impact', ascending=False)
    
    print(df_coefs)
    
    print("\n💡 Interprétation :")
    print("- Un coef POSITIF augmente la PD (plus risqué).")
    print("- Un coef NÉGATIF diminue la PD (plus sûr).")
    print("- Note : Les variables caté sont transformées en WoE avant d'arriver ici.")

if __name__ == "__main__":
    inspect()