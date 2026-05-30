from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


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

        # Define machine learning models
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

        # Store trained models
        self.trained_models = {}

        # Store model evaluation results
        self.results = {}

        # Create label encoder for target variable
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df):
        '''
        Prepares the dataset for machine learning by separating the target variable,
        one-hot encoding categorical features, and label encoding the target.

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            tuple: Feature matrix (X) and target vector (y)
        '''

        # Separate features and target column
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        # Convert categorical features to numerical values
        X = pd.get_dummies(X)

        # Encode target labels
        y = self.label_encoder.fit_transform(y)

        return X, y

    def split_data(self, X, y):
        '''
        Splits the dataset into training and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target values

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        '''

        # Display split status
        print("Training and Validation dataset completed")

        # Split data into training and testing sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scale_data(self, X_train, X_test):
        '''
        Standardizes feature values for model training.
    
        Args:
            X_train: Training features
            X_test: Testing features
    
        Returns:
            tuple: Scaled training and testing features
        '''
    
        print("Scaling features...")
    
        # Create scaler object
        scaler = StandardScaler()
    
        # Scale training data
        X_train = scaler.fit_transform(X_train)
    
        # Scale testing data
        X_test = scaler.transform(X_test)
    
        print("Feature scaling completed")
        
        return X_train, X_test

    def train_models(self, X_train, y_train):
        '''
        Trains all configured machine learning models.

        Args:
            X_train: Training features
            y_train: Training labels
        '''

        print("Training machine learning models...")
        
        # Train and store each model
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
        print("All models trained successfully")

    def evaluate_models(self, X_test, y_test):
        '''
        Evaluates all trained models using test data.

        Args:
            X_test: Testing features
            y_test: Actual testing labels
        '''
        
        print("Freddy Fazbear Presents")
        
        # Evaluate each trained model
        for name, model in self.trained_models.items():
    
            print(f"Evaluating {name}...")
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            self.results[name] = acc
            print("-----------------------")
            print(name)
            print("Accuracy:", acc)
            print(classification_report(y_test, preds))
    
        print("Model evaluation completed")

    def run(self, df):
        '''
        Executes the complete machine learning workflow.

        Args:
            df (pd.DataFrame): Input dataset
        '''
        
        print("--------------------------")

        # Prepare data for training
        X, y = self.prepare_data(df)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale feature values
        X_train, X_test = self.scale_data(X_train, X_test)

        # Train machine learning models
        self.train_models(X_train, y_train)

        # Evaluate trained models
        self.evaluate_models(X_test, y_test)

        print("-------------------------")
        print("Machine learning completed")