from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class ModelTrainer:

    def __init__(self, target_col):
        self.target_col = target_col
        # move all these into config later
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

            "XGBoost": XGBClassifier(
                eval_metric="logloss"
            ),

            "SVM (RBF)": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            )
        }

        self.trained_models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df):
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        X = pd.get_dummies(X)
        y = self.label_encoder.fit_transform(y)
        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def train_models(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate_models(self, X_test, y_test):
        for name, model in self.trained_models.items():
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            self.results[name] = acc
            print("-----------------------")
            print(name)
            print("Accuracy:", acc)
            print(classification_report(y_test, preds))

    def run(self, df):
        print("Training models")
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train, X_test = self.scale_data(X_train, X_test)
        self.train_models(X_train, y_train)
        self.evaluate_models(X_test, y_test)