import pandas as pd

class FeatureEngineer:
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