import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    '''
    Reads a yaml file and returns a dictionary

    Args:
        config_name: Config file to be read

    Returns:
        dict: The loaded configuration

    Raises:
        ValueError: If config file is not found
    '''
    config_path = Path(__file__).parent / config_name
    # Check if config file exists
    if not config_path.exists():
        # Raise an error and stop if config file not found
        raise ValueError(f"Configuration file not found at: {config_path.absolute()}")
    
    # Load configs and return as dictionary
    with open(config_path, "r") as f:
        return yaml.safe_load(f)