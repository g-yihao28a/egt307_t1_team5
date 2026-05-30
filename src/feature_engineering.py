import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureEngineer:

    def __init__(self):

        self.numeric_features = [
            "temperature",
            "humidity",
            "co2_infraredsensor",
            "co2_electrochemicalsensor",
            "metaloxidesensor_unit1",
            "metaloxidesensor_unit2",
            "metaloxidesensor_unit3",
            "metaloxidesensor_unit4"
        ]

    def process(self, df: pd.DataFrame):

        df = df.copy()

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            df[self.numeric_features]
        )

        pca = PCA(n_components=3)

        pca_features = pca.fit_transform(
            scaled_data
        )

        df["PC1"] = pca_features[:, 0]
        df["PC2"] = pca_features[:, 1]
        df["PC3"] = pca_features[:, 2]

        print(
            f"PCA explained variance: "
            f"{pca.explained_variance_ratio_.sum():.2%}"
        )

        print("Feature engineering complete")

        return df