from imblearn.over_sampling import SMOTE
from config import load_config

config = load_config("config.yaml")
random_state = config["random"]

def generate_data(X_train, y_train):
    """
        Takes training data and creates more examples for the minority classes through interpolation using KNN
    """
    print("Generating synthetic data")
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train