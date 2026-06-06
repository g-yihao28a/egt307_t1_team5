import pandas as pd
from sqlalchemy import create_engine, inspect, text
from pathlib import Path
import sqlalchemy
from config import load_config

# Load Config
config = load_config("config.yaml")
data_cfg = config["data"]

db_path = Path.cwd() / data_cfg["data_directory"] / data_cfg["db_name"]
table_name = data_cfg["table_name"]

def load_data_from_db(db_path=db_path, table_name=table_name):
    '''
    Reads data from a SQL database and returns it as a DataFrame

    Args:
        db_path (Path): The path to the SQLite database file
        table_name (str): The name of the table to load

    Returns:
        pd.DataFrame: The loaded data

    Raises:
        FileNotFoundError: If the database path does not exist
        ValueError: If the table does not exist in the database
    '''
    # Check if path exists
    if not db_path.exists():
        # Raise an error to stop if path does not exist
        raise FileNotFoundError(f"Database file missing: {db_path}")
    
    # Create engine to load db file
    engine = create_engine(f"sqlite:///{db_path.resolve()}")
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Check table exists
    if table_name not in table_names:
        # Raise an error to stop if table not found
        raise ValueError(f"Target table '{table_name}' not found. Available tables: {table_names}")
    
    # Load data into dataframe and return it
    print(f"Target table: {table_name} found\nLoading data")
    return pd.read_sql_table(table_name, engine)