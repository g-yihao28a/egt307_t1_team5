from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import json
import os
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from datetime import datetime

class ModelTrainer:
    '''
    Handles seperating data into training and validation sets, model training and evaluation
    for multiple machine learning classifiers.
    '''

    def __init__(self, target_col):
        '''
        Initializes the ModelTrainer class.

        Args:
            target_col (str): The target column to predict
        '''

        # Store target column name
        self.target_col = target_col

        # Define hyperparameter grids
        self.param_distributions = {
            "Random Forest": {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 10)
            },
            "XGBoost": {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10)
            },
            "SVM (RBF)": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
        
        # Define machine learning models
        self.models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "SVM (RBF)": SVC(class_weight="balanced")
        }

        # Create label encoder for target variable
        self.label_encoder = LabelEncoder()

    def train_models(self, X_train, y_train):
        '''
        Trains and optimises all configured machine learning models.

        Args:
            X_train: Training features
            y_train: Training labels
        '''
        y_train = self.label_encoder.fit_transform(y_train)
        self.best_models = {}
        print("Training machine learning models...")
        
        # Train and store each model
        for name, model in self.models.items():
            print(f"Optimising {name}...")
            search = RandomizedSearchCV(
                model, 
                param_distributions=self.param_distributions[name],
                n_iter=10, # Number of parameter settings to try
                cv=3,      # 3-fold cross-validation
                n_jobs=-1, # Use all CPU cores
                random_state=42
            )
            search.fit(X_train, y_train)
            self.best_models[name] = search.best_estimator_
            print(f"Best params for {name}: {search.best_params_}")
            
        print("All models trained successfully")

    def evaluate_models(self, X_test, y_test):
        '''
        Evaluates all trained models using test data.

        Args:
            X_test: Testing features
            y_test: Actual testing labels
        '''
        y_test = self.label_encoder.transform(y_test)

        # Store model evaluation results
        self.results = {}
        # Evaluate each trained model
        for name, model in self.best_models.items():
    
            print(f"Evaluating {name}...")
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            self.results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average='weighted'),
                "recall": recall_score(y_test, preds, average='weighted'),
                "f1_score": f1_score(y_test, preds, average='weighted')
            }
            
            print(f"Results for {name}: {self.results[name]}")
            print("--------------------------")
            print(classification_report(y_test, preds, target_names=self.label_encoder.classes_))
    
        print("Model evaluation completed")

    def export_results(self, filename="experiment_log.csv"):
            """
            Appends the results and best parameters to a single CSV summary file.
            """
            # Prepare data for a single row
            # Flattening results for the CSV structure
            row_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add accuracy/metrics for each model
            for name, metrics in self.results.items():
                # If metrics is a dict, flatten it
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        row_data[f"{name}_{metric_name}"] = value
                else:
                    # Fallback for simple accuracy float
                    row_data[f"{name}_accuracy"] = metrics
            
            # Add best parameters (as a string to keep the CSV clean)
            params_str = {name: str(model.get_params()) for name, model in self.best_models.items()}
            row_data["parameters"] = str(params_str)

            # Convert to DataFrame and append
            df = pd.DataFrame([row_data])
            
            # Write to CSV (create if it does not exist, otherwise append)
            file_exists = os.path.isfile(filename)
            df.to_csv(filename, mode='a', index=False, header=not file_exists)
            
            print(f"Results appended to '{filename}'.")

    def run(self, X_train, y_train, X_test, y_test):
        '''
        Executes the complete machine learning workflow.

        Args:
            df (pd.DataFrame): Input dataset
        '''
        
        print("--------------------------")

        # Train machine learning models
        self.train_models(X_train, y_train)

        # Evaluate trained models
        self.evaluate_models(X_test, y_test)

        print("-------------------------")
        print("Machine learning completed")