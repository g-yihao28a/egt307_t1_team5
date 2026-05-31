import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, target_col):
        self.categorical_features = [
            'time_of_day',
            'co_gassensor',
            'hvac_operation_mode',
            'ambient_light_level',
            'activity_level'
            ]
        
        self.numeric_features = [
            'temperature',
            'humidity',
            'co2_infraredsensor',
            'co2_electrochemicalsensor',
            'metaloxidesensor_unit1',
            'metaloxidesensor_unit2',
            'metaloxidesensor_unit3',
            'metaloxidesensor_unit4'
            ]
        self.target_col = target_col
        self.features_to_use_numeric = [c for c in self.numeric_features if c != self.target_col]
        self.features_to_use_categoric = [c for c in self.categorical_features if c != self.target_col]

        self.pipeline = None
        self.scale_and_encode()

    def scale_and_encode(self):
        # Define routes for data types
        numeric_transformer = StandardScaler()
        categoric_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Combine into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.features_to_use_numeric),
                ('cat', categoric_transformer, self.features_to_use_categoric)
            ]
        )
        # Pipeline for preparation
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        self.pipeline.set_output(transform="pandas")

    def process(self, df):

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit and transform
        X_train = self.pipeline.fit_transform(X_train)
        X_test = self.pipeline.transform(X_test)
        
        print("Preprocessing pipeline complete.")
        return X_train, X_test, y_train, y_test