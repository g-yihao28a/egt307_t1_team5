import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureEngineer:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame):
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        self.pca.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("You must call fit() before transform().")
        
        # Select only numeric-looking columns for PCA
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        pca_features = self.pca.transform(X[numeric_cols])
        
        # Create a DataFrame for the PCs to keep it compatible with pandas
        pc_df = pd.DataFrame(
            pca_features, 
            columns=[f"PC{i+1}" for i in range(pca_features.shape[1])],
            index=X.index
        )
        
        # Merge back with the rest of the data
        return pd.concat([X, pc_df], axis=1)