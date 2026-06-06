import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from config import load_config

config = load_config("config.yaml")
clean_cfg = config["clean"]

columns_to_drop = clean_cfg["columns_to_drop"]
temperature_threshold = clean_cfg["temperature_threshold"]
min_humidity, max_humidity = clean_cfg["min_humidity"], clean_cfg["max_humidity"]
optimised_n = clean_cfg["optimised_n"]
activity_map = clean_cfg["activity_map"]
activity_map = {k: v for d in activity_map for k, v in d.items()}
categorical_columns = clean_cfg["categorical_columns"]
numerical_columns = clean_cfg["numerical_columns"]

class DataCleaner:
    def __init__(self):
        # move all these into config later
        self.columns_to_drop = columns_to_drop
        self.temperature_threshold = temperature_threshold
        self.min_humidity, self.max_humidity = min_humidity, max_humidity # Assuming percentage
        self.optimised_n = optimised_n
        self.activity_map = activity_map # Activity map to standardise "activity level" data
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def drop_columns(self, df: pd.DataFrame):
        """
        Drops specified columns from the dataframe.
        """
        # Use errors='ignore' to avoid crashing if the column is already gone
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        return df

    def standardise_column_names(self, df: pd.DataFrame):
        """
        Converts column names to lowercase and replaces spaces with underscores
        Example: "Time of Day" -> "time_of_day"
        """
        df.columns = (
            df.columns
            .str.strip()             # Remove leading/trailing spaces
            .str.lower()             # Convert to lowercase
            .str.replace(' ', '_')   # Replace spaces with underscores
        )
        return df

    def standardise_hvac(self, df:pd.DataFrame):
        """
        Standardizes HVAC values to lowercase
        """
        if "hvac_operation_mode" in df.columns:
            df["hvac_operation_mode"] = df["hvac_operation_mode"].str.lower().str.strip()
        return df
    
    def standardise_activity(self, df:pd.DataFrame):
        """
        Standardises naming of activity level values
        Example: "LowActivity" -> "low_activity"
        """
        if "activity_level" in df.columns:
            df["activity_level"] = df["activity_level"].replace(self.activity_map)
        return df
    
    def drop_duplicates(self, df:pd.DataFrame):
        """
        Removes duplicate rows and resets the index to match the updated length of the dataframe
        """
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        return df
    
    def convert_kelvin_to_celsius(self, df:pd.DataFrame):
        """
        Converts likely Kelvin values (above threshold) and converts to Celsius
        """
        df.loc[df['temperature'] > self.temperature_threshold, 'temperature'] -= 273.15  
        return df
    
    def remove_outlier_humidity_values(self, df: pd.DataFrame):
        """
        Sets humidity values outside the specified range to NaN to be imputed later
        """
        df.loc[(df["humidity"] > self.max_humidity) | (df["humidity"] < self.min_humidity), "humidity"] = np.nan
        return df

    def convert_co_gassensor_to_string(self, df: pd.DataFrame):
        """
        Converts CO Gas Sensor column to string, keeping NaN values as NaN.
        """
        df["co_gassensor"] = df["co_gassensor"].astype("string")
        df.loc[df["co_gassensor"] == "nan", "co_gassensor"] = np.nan
        return df
    
    def impute_missing_data(self, df:pd.DataFrame):
        """
        Use KNN model to impute missing data
        """
        # Only scale numerical columns
        scaler = MinMaxScaler()
        df_num = df[self.numerical_columns].copy()
        df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=self.numerical_columns, index=df.index)

        # Encode categoricals
        encoders = {}
        df_cat = df[self.categorical_columns].copy()
        for col in self.categorical_columns:
            df_cat[col] = df_cat[col].astype(object)
            enc = OrdinalEncoder()
            mask = df_cat[col].notnull()
            df_cat.loc[mask, [col]] = enc.fit_transform(df_cat.loc[mask, [col]])
            encoders[col] = enc
        
        # Concatenate back for imputation
        df_combined = pd.concat([df_num_scaled, df_cat], axis=1)

        # Impute
        imputer = KNNImputer(n_neighbors=self.optimised_n, weights='distance')
        df_imputed = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns, index=df.index)

        # Inverse scale numerical
        df_imputed[self.numerical_columns] = scaler.inverse_transform(df_imputed[self.numerical_columns])

        # Decode categorical
        for col in self.categorical_columns:
            df_imputed[col] = encoders[col].inverse_transform(df_imputed[[col]].round().astype(int)).flatten()
            
        return df_imputed
                            
    def process(self, df):
        """
        Runs through entire preprocessing pipeline
        Pipeline includes standardisation, deduplication, outlier removal, and KNN imputation.
        
        Args:
            df (pd.DataFrame): The raw input dataset
            
        Returns:
            pd.DataFrame: The fully processed, clean dataset with 
                          missing values filled via KNN
        """

        print("Starting cleaning")
        df = df.copy()
        df = self.standardise_column_names(df)
        df = self.drop_columns(df)
        df = self.standardise_hvac(df)
        df = self.standardise_activity(df)
        df = self.drop_duplicates(df)

        df = self.convert_kelvin_to_celsius(df) # Maybe try to check whether values are truly in kelvin
        df = self.remove_outlier_humidity_values(df)
        #df = self.convert_co_gassensor_to_string(df) Not needed anymore because column has been dropped

        df = self.impute_missing_data(df)
        print("Done cleaning")
        return df