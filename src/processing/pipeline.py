from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.processing.woe_encoder import WoeEncoder

def create_scoring_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """
    Creates the full training pipeline for the Scorecard.
    
    Pipeline architecture:
    1. Numeric branch: Imputation (Median) -> Standardization (Scaling)
    2. Categorical branch: Imputation (Most frequent) -> WoE Encoding
    3. Model: Logistic Regression
    
    Args:
        categorical_features: List of categorical column names (e.g., 'sector', 'rating_agency').
        numeric_features: List of numeric column names (e.g., 'leverage_ratio', 'years_in_business').
        
    Returns:
        Untrained Scikit-Learn Pipeline.
    """
    
    # 1. Processing numeric variables
    # Fill missing values with the median and scale features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Processing categorical variables
    # Fill missing values with 'MISSING' then apply WoE encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('woe', WoeEncoder(columns=None))  # Columns passed through ColumnTransformer
    ])

    # 3. Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Final pipeline with model
    # class_weight='balanced' is crucial because defaults are rare (Imbalanced Dataset)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', random_state=42))
    ])

    return model

def extract_pd_from_proba(proba_array):
    """
    Helper to cleanly extract PD from predict_proba output.
    Class 1 corresponds to default.
    """
    # predict_proba returns [[prob_0, prob_1], ...]
    return proba_array[:, 1]
