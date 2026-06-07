from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import os
from datetime import datetime


class Evaluation:

    def __init__(self, best_models, label_encoder, X_train, X_test, y_test):
        """
        Initializes the Evaluation class.

        Args:
            best_models: Dictionary of trained machine learning models
            label_encoder: Encoder used to transform target labels
            X_train: Training features (used for feature importance)
            X_test: Testing features
            y_test: Actual test labels
        """
        self.best_models = best_models
        self.label_encoder = label_encoder
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_models(self):
        """
        Evaluates all trained models using test data.

        Args:
            X_test: Testing features
            y_test: Actual testing labels
        """

        X_test = self.X_test
        y_test = self.label_encoder.transform(self.y_test)

        # Store model evaluation results
        self.results = {}

        # Evaluate each trained model
        for name, model in self.best_models.items():

            print(f"Evaluating {name}...")

            preds = model.predict(X_test)

            # Compute evaluation metrics
            self.results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average='weighted'),
                "recall": recall_score(y_test, preds, average='weighted'),
                "f1_score": f1_score(y_test, preds, average='weighted')
            }

            print(f"Results for {name}: {self.results[name]}")
            print("--------------------------")

            print(classification_report(
                y_test,
                preds,
                target_names=self.label_encoder.classes_
            ))

            cm = confusion_matrix(y_test, preds)
            print(cm)

            # Feature importance (ONLY works for tree-based models)
            if hasattr(model, "feature_importances_"):

                importances = model.feature_importances_
                feature_names = self.X_train.columns

                # Create ranking of feature importance
                feat_imp = pd.Series(
                    importances,
                    index=feature_names
                ).sort_values(ascending=False)

                print(feat_imp.head(100))

        print("Model evaluation completed")

        return self.results

    def export_results(self, filename="experiment_log.csv"):
        """
        Appends the results and best parameters to a single CSV summary file.

        This function:
        - Flattens evaluation metrics
        - Saves model performance
        - Stores hyperparameters used
        """

        # Prepare data for a single row
        # Flattening results for CSV structure
        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add accuracy/precision/recall/f1 for each model
        for name, metrics in self.results.items():

            # If metrics is a dictionary, flatten it
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    row_data[f"{name}_{metric_name}"] = value

            else:
                # Fallback for simple accuracy float
                row_data[f"{name}_accuracy"] = metrics

        # Store best model hyperparameters as string (for logging)
        params_str = {
            name: str(model.get_params())
            for name, model in self.best_models.items()
        }

        row_data["parameters"] = str(params_str)

        # Convert to DataFrame
        df = pd.DataFrame([row_data])

        # Check if file already exists
        file_exists = os.path.isfile(filename)

        # Append results to CSV
        df.to_csv(
            filename,
            mode='a',
            index=False,
            header=not file_exists
        )

        print(f"Results appended to '{filename}'.")