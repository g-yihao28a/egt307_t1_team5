import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, activity_map=None):
        # move all these into config later
        self.columns_to_drop = ["session_id"]
        self.temperature_threshold = 50
        self.min_humidity, self.max_humidity = 0, 100 # Assuming percentage
        self.optimised_n = 5
        # Activity map to standardise "activity level" data
        self.activity_map = activity_map or {
            'Low Activity': 'low_activity',
            'LowActivity': 'low_activity',
            'Low_Activity': 'low_activity',
            'Moderate Activity': 'moderate_activity',
            'ModerateActivity': 'moderate_activity',
            'High Activity': 'high_activity',
        }
        self.categorical_columns = [
            'time_of_day',
            'co_gassensor',
            'hvac_operation_mode',
            'ambient_light_level',
            'activity_level'
            ]
        
        self.numerical_columns = [
            'temperature',
            'humidity',
            'co2_infraredsensor',
            'co2_electrochemicalsensor',
            'metaloxidesensor_unit1',
            'metaloxidesensor_unit2',
            'metaloxidesensor_unit3',
            'metaloxidesensor_unit4'
            ]

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
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import MinMaxScaler
        # Encoding - KNN requires alla imputers to be numeric
        encoders = {}
        for col in self.categorical_columns:
            encoder = OrdinalEncoder()
            # Mask non-null values
            mask = df[col].notnull()
            # Convert data type of categorical columns to hold objects temporarily (numbers and letters)
            df[col] = df[col].astype(object) 
            
            # Fit_transform encodes the categorical data e.g. "morning":1, "afternoon":2, etc..
            # df.loc[mask, [col]] extracts masked values as a dataframe which is required
            encoded_vals = encoder.fit_transform(df.loc[mask, [col]])
            
            # df.loc[mask, col] is the data just transformed, 
            # flatten() changes the values back from a datafrom to a series to put back into column
            df.loc[mask, col] = encoded_vals.flatten()
            
            # Change back to numeric, keeping NaNs as NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save encoding dictionary for decoding after imputation
            encoders[col] = encoder

        # Scaling
        # Ensure numerical columns are float, not int, and scales values to between 0 and 1
        scaler = MinMaxScaler()
        df[self.numerical_columns] = scaler.fit_transform(df[self.numerical_columns].astype(float))

        #Imputation
        # weights='distance' prioritises closer neighbours instead of n_neighbours having equal weight
        imputer = KNNImputer(n_neighbors=self.optimised_n, weights='distance')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df), 
            columns=df.columns, 
            index=df.index
        )

        # Reverse scale numerical columns
        df_imputed[self.numerical_columns] = scaler.inverse_transform(df_imputed[self.numerical_columns])

        # Decode categorical columns back
        for col in self.categorical_columns:
            # Use .round() to round float values generated to integer to be decoded
            df_imputed[col] = encoders[col].inverse_transform(
                df_imputed[[col]].round().astype(int)
            ).flatten()
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

        print("Starting preprocessing") # Maybe change this
        df = df.copy()
        df = self.standardise_column_names(df)
        df = self.drop_columns(df) # Drop Session ID
        df = self.standardise_hvac(df)
        df = self.standardise_activity(df)
        df = self.drop_duplicates(df)

        df = self.convert_kelvin_to_celsius(df) # Maybe try to check whether values are truly in kelvin
        df = self.remove_outlier_humidity_values(df)
        df = self.convert_co_gassensor_to_string(df)
        df = self.impute_missing_data(df)
        print("Done preprocessing")
        return df