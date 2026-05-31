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

    # Split Data
    preprocessor = DataPreprocessor(target_col="activity_level")
    X_train, X_test, y_train, y_test = preprocessor.process(df)

    # Feature Engineering
    engineer = FeatureEngineer()
    engineer.fit(X_train)
    X_train = engineer.transform(X_train)
    X_test = engineer.transform(X_test)

    # Train Model
    trainer = ModelTrainer("activity_level")
    trainer.run(X_train, y_train, X_test, y_test)
    trainer.export_results()

    # Export Results
except ValueError as e:
    print(f"Aborting pipeline: {e}")
    exit(1)