import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureEngineer:
    def __init__(self
                 #, n_components=0.95
                 ):
    #     self.pca = PCA(n_components=n_components)
    #     self.is_fitted = False
        pass

    def add_features(self, df):
        df = df.copy()

        # Ratio (Adding a small epsilon to avoid division by zero)
        df['CO2_VOC_Ratio'] = df['co2_electrochemicalsensor'] / (df['metaloxidesensor_unit2'] + 1e-6)
        
        # Variability (Spread across units)
        sensor_cols = ['metaloxidesensor_unit1', 'metaloxidesensor_unit2', 
                       'metaloxidesensor_unit3', 'metaloxidesensor_unit4']
        df['metal_oxide_std'] = df[sensor_cols].std(axis=1)
        
        # Total Load
        df['total_load'] = df[['co2_electrochemicalsensor', 'metaloxidesensor_unit2']].sum(axis=1)
        
        return df

    # def fit(self, X_train: pd.DataFrame):
    #     numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    #     self.pca.fit(X_train)
    #     self.is_fitted = True
    #     return self

    # def transform(self, X: pd.DataFrame):
    #     if not self.is_fitted:
    #         raise RuntimeError("You must call fit() before transform().")
        
    #     # Select only numeric-looking columns for PCA
    #     numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    #     pca_features = self.pca.transform(X[numeric_cols])
        
    #     # Create a DataFrame for the PCs to keep it compatible with pandas
    #     pc_df = pd.DataFrame(
    #         pca_features, 
    #         columns=[f"PC{i+1}" for i in range(pca_features.shape[1])],
    #         index=X.index
    #     )
        
    #     # Merge back with the rest of the data
    #     return pd.concat([X, pc_df], axis=1)