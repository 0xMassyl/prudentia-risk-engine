import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Any

class WoeEncoder(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé Scikit-Learn pour l'encodage Weight of Evidence (WoE).
    """

    def __init__(self, columns: list[str] | None = None, regularization: float = 1.0):
        self.columns = columns
        self.regularization = regularization
        self.mapping_: dict[str, dict[str | float, float]] = {}
        self.iv_: dict[str, float] = {}

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        """S'assure que l'entrée est un DataFrame, même si c'est un NumPy array."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if self.columns is not None:
                return pd.DataFrame(X, columns=self.columns)
            return pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        raise TypeError(f"Input type non supporté: {type(X)}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcule les WoE pour chaque colonne spécifiée.
        """
        X = self._ensure_dataframe(X)
        
        if self.columns is None:
             self.columns = X.columns.tolist()

        total_bads = y.sum()
        total_goods = y.count() - total_bads
        
        if total_bads == 0 or total_goods == 0:
            raise ValueError("Le target 'y' doit contenir à la fois des classes 0 et 1.")

        for col in self.columns:
            df_calc = pd.DataFrame({'feature': X[col], 'target': y})
            
            stats = df_calc.groupby('feature', dropna=False)['target'].agg(['count', 'sum'])
            stats.columns = ['total', 'bads']
            stats['goods'] = stats['total'] - stats['bads']
            
            dist_goods = (stats['goods'] + self.regularization) / (total_goods + 2 * self.regularization)
            dist_bads = (stats['bads'] + self.regularization) / (total_bads + 2 * self.regularization)
            
            woe_values = np.log(dist_goods / dist_bads)
            
            iv_contribution = (dist_goods - dist_bads) * woe_values
            self.iv_[col] = float(iv_contribution.sum())
            
            # --- CORRECTION DU DICTIONNAIRE ---
            d = woe_values.to_dict()
            
            # On cherche s'il y a déjà une clé qui est considérée comme NaN (np.nan, float('nan'), None...)
            nan_keys = [k for k in d.keys() if pd.isna(k)]
            
            if nan_keys:
                # Si un NaN existe déjà (ex: généré par groupby), on normalise la clé à np.nan
                # On prend la valeur du premier NaN trouvé
                val = d[nan_keys[0]]
                # On supprime toutes les variations de NaN pour éviter les doublons
                for k in nan_keys:
                    del d[k]
                # On réinsère proprement
                d[np.nan] = val
            else:
                # Si aucun NaN n'a été vu dans l'entraînement, on fixe la valeur par défaut à 0.0
                d[np.nan] = 0.0
            
            self.mapping_[col] = d

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les catégories par leur valeur WoE.
        """
        check_is_fitted(self, 'mapping_')
        X_out = self._ensure_dataframe(X)

        if self.columns is None:
             return X_out

        for col in self.columns:
            if col in X_out.columns:
                mapped_series = X_out[col].map(self.mapping_[col])
                X_out[col] = mapped_series.fillna(0.0).astype(float)
                
        return X_out