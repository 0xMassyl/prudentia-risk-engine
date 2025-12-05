from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.processing.woe_encoder import WoeEncoder

def create_scoring_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """
    Crée le pipeline complet d'apprentissage pour le Scorecard.
    
    Architecture du Pipeline :
    1. Branche Numérique : Imputation (Moyenne) -> Standardisation (Scalage)
    2. Branche Catégorielle : Imputation (Mode) -> Encodage WoE
    3. Modèle : Régression Logistique
    
    Args:
        categorical_features: Liste des noms de colonnes catégorielles (ex: 'sector', 'rating_agency').
        numeric_features: Liste des noms de colonnes numériques (ex: 'leverage_ratio', 'years_in_business').
        
    Returns:
        Pipeline Scikit-Learn non entraîné.
    """
    
    # 1. Traitement des variables numériques
    # On remplit les trous par la médiane et on met à l'échelle
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Traitement des variables catégorielles
    # On remplit les trous par 'MISSING' puis on transforme en WoE
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('woe', WoeEncoder(columns=None)) # Les colonnes sont passées par le ColumnTransformer
    ])

    # 3. Assemblage du préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Pipeline final avec le modèle
    # class_weight='balanced' est crucial car les défauts sont rares (Imbalanced Dataset)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', random_state=42))
    ])

    return model

def extract_pd_from_proba(proba_array):
    """
    Helper pour extraire proprement la PD du résultat de predict_proba.
    La classe 1 est le défaut.
    """
    # predict_proba renvoie [[prob_0, prob_1], ...]
    return proba_array[:, 1]