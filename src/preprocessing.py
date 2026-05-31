import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, target_col):
        self.target_col = target_col
        self.pipeline = None

    def _auto_identify_features(self, df):
        # Drop the target from the dataframe to identify only features
        features = df.drop(columns=[self.target_col])
        
        # Automatically identify numeric columns (float, int)
        self.numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Automatically identify categorical columns (object, category, bool)
        self.categorical_features = features.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        print(f"Detected Numeric: {self.numeric_features}")
        print(f"Detected Categorical: {self.categorical_features}")

    def scale_and_encode(self):
        numeric_transformer = StandardScaler()
        categoric_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categoric_transformer, self.categorical_features)
            ]
        )
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        self.pipeline.set_output(transform="pandas")

    def process(self, df):
        # Identify types dynamically based on the input dataframe
        self._auto_identify_features(df)
        self.scale_and_encode()
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = self.pipeline.fit_transform(X_train)
        X_test = self.pipeline.transform(X_test)
        
        return X_train, X_test, y_train, y_test