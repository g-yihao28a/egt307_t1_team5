from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import json
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
from datetime import datetime
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class ModelTrainer:
    '''
    Handles seperating data into training and validation sets, model training and evaluation
    for multiple machine learning classifiers.
    '''

    def __init__(self, target_col, X_train, y_train, X_test, y_test):
        '''
        Initializes the ModelTrainer class.

        Args:
            target_col (str): The target column to predict
        '''

        # Store target column name
        self.target_col = target_col
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        #self.sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Define hyperparameter grids
        self.param_distributions = {
            "Random Forest": {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': randint(2, 21)
            },
            "XGBoost": {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': [0.7, 0.8, 0.9, 1.0], 
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            "SVM (RBF)": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
        }

        # Define machine learning models
        self.models = {
            "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
            "XGBoost": XGBClassifier(objective='multi:softprob', eval_metric="mlogloss", num_class=len(np.unique(self.y_train)), random_state=42),
            "SVM (RBF)": SVC(class_weight="balanced"),
        }

        # Create label encoder for target variable
        self.label_encoder = LabelEncoder()

    def train_models(self):
        '''
        Trains and optimises all configured machine learning models.

        Args:
            X_train: Training features
            y_train: Training labels
        '''
        # Split into train and validation for early stopping
        y_encoded = self.label_encoder.fit_transform(self.y_train)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train, y_encoded, 
            test_size=0.1, stratify=y_encoded, random_state=42
        )
        # Recalculate weights on this subset
        self.sample_weights_sub = compute_sample_weight(
            class_weight='balanced', 
            y=y_train_sub
        )
        

        self.best_models = {}
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        print("Training machine learning models...")
        
        # Train and store each model
        for name, model in self.models.items():
            print(f"Optimising {name}...")
            fit_params = {}
            if name == "XGBoost":
                model.set_params(early_stopping_rounds=5)
                fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}

            search = RandomizedSearchCV(
                    model, 
                    param_distributions=self.param_distributions[name],
                    n_iter=25, # Number of parameter settings to try
                    cv=cv_strategy,      # 3-fold cross-validation
                    scoring="f1_weighted",
                    n_jobs=-1, # Use all CPU cores
                    random_state=42
                )

            search.fit(
                X_train_sub, y_train_sub, 
                sample_weight=self.sample_weights_sub,
                **fit_params
            )

            self.best_models[name] = search.best_estimator_
            print(f"Best params for {name}: {search.best_params_}")
            
        print("All models trained successfully")

    def evaluate_models(self):
        '''
        Evaluates all trained models using test data.

        Args:
            X_test: Testing features
            y_test: Actual testing labels
        '''
        X_test = self.X_test
        y_test = self.label_encoder.transform(self.y_test)

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
            cm = confusion_matrix(y_test, preds)
            print(cm)

            importances = self.best_models['Random Forest'].feature_importances_
            feature_names = self.X_train.columns

            # Create a ranking
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            print(feat_imp.head(100))
    
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

    def run(self):
        '''
        Executes the complete machine learning workflow.

        Args:
            df (pd.DataFrame): Input dataset
        '''
        
        print("--------------------------")

        # Train machine learning models
        self.train_models()

        # Evaluate trained models
        self.evaluate_models()

        print("-------------------------")
        print("Machine learning completed")