'''
main.py: Main file to be ran
Program Flow:
- Load Config
- Load Data
- Clean Data (class)
- Train Model (class)
- Export Results
'''
from pathlib import Path
from config import load_config
from ingestion import load_data_from_db
from cleaning import DataCleaner
from training import ModelTrainer
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor

from imblearn.over_sampling import SMOTE

try:
    # Load Config
    config = load_config("config.yaml")
    data_cfg = config["data"]

    # Load Data
    db_path = Path.cwd() / data_cfg["data_directory"] / data_cfg["db_name"]
    table_name = data_cfg["table_name"]
    df = load_data_from_db(db_path, table_name)

    # Clean Data
    cleaner = DataCleaner()
    df = cleaner.process(df)

    # Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.add_features(df)
    
    # Split Data
    preprocessor = DataPreprocessor(target_col="activity_level")
    X_train, X_test, y_train, y_test = preprocessor.process(df)

    # Synthetic data generation (Move into another file later)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train Model
    trainer = ModelTrainer("activity_level", X_train, y_train, X_test, y_test)
    trainer.run()
    trainer.export_results()

    # Export Results
except ValueError as e:
    print(f"Aborting pipeline: {e}")
    exit(1)