import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "data/models/scorecard_model.pkl"

def inspect():
    print(f" Loading model from {MODEL_PATH}...")
    
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Access the Logistic Regression model (last step of the pipeline)
    log_reg = pipeline.named_steps['classifier']
    
    print("\n --- INTERCEPT & COEFFICIENTS ---")
    print(f"Intercept (Base bias): {log_reg.intercept_[0]:.4f}")
    
    # Retrieving feature names (slightly technical due to ColumnTransformer)
    # We assume the order: Numerical first, then Categorical
    numeric_features = ['age', 'income', 'loan_amount', 'years_employed']
    categorical_features = ['sector', 'rating_agency', 'region']
    feature_names = numeric_features + categorical_features
    
    coeffs = log_reg.coef_[0]
    
    df_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coeffs,
        'Impact': np.abs(coeffs)  # Used to sort by importance
    }).sort_values('Impact', ascending=False)
    
    print(df_coefs)
    
    print("\n Interpretation:")
    print("- A POSITIVE coefficient increases PD (riskier).")
    print("- A NEGATIVE coefficient reduces PD (safer).")
    print("- Note: Categorical variables are transformed via WoE before reaching the model.")

if __name__ == "__main__":
    inspect()
