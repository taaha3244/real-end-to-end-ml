import os
import sys
import pickle
# Directly set the project root directory
project_root = "/workspaces/real-end-to-end-ml"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logging.custom_logger import logger
from src.custom_exceptions import CustomException

class ModelCreation:
    """
    A class to create and log machine learning models using MLflow.

    Attributes
    ----------
    data_path : str
        The path to the directory containing the preprocessed data.
    X_train : pd.DataFrame
        The training features.
    X_test : pd.DataFrame
        The testing features.
    y_train : pd.Series
        The training target variable.
    y_test : pd.Series
        The testing target variable.
    model : RandomForestClassifier
        The machine learning model.

    Methods
    -------
    load_preprocessed_data():
        Loads the preprocessed training and testing data.
    
    train_model():
        Trains the machine learning model.
    
    evaluate_model():
        Evaluates the machine learning model.
    
    log_model():
        Logs the model, parameters, and metrics to MLflow.
    
    save_model():
        Saves the model locally.
    """

    def __init__(self, data_path, local_model_path):
        """
        Constructs all the necessary attributes for the ModelCreation object.

        Parameters
        ----------
        data_path : str
            The path to the directory containing the preprocessed data.
        local_model_path : str
            The local path to save the trained model.
        """
        self.data_path = data_path
        self.local_model_path = local_model_path
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_preprocessed_data()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_preprocessed_data(self):
        """
        Loads the preprocessed training and testing data.

        Returns
        -------
        tuple
            The preprocessed training and testing features and target variables.

        Raises
        ------
        CustomException
            If there is an error loading the preprocessed data.
        """
        try:
            X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(self.data_path, 'X_test.csv'))
            y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'))
            y_test = pd.read_csv(os.path.join(self.data_path, 'y_test.csv'))

            logger.info('Preprocessed data loaded successfully')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_model(self):
        """
        Trains the machine learning model.

        Raises
        ------
        CustomException
            If there is an error during model training.
        """
        try:
            self.model.fit(self.X_train, self.y_train.values.ravel())
            logger.info('Model trained successfully')
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(self):
        """
        Evaluates the machine learning model.

        Returns
        -------
        dict
            The evaluation metrics.

        Raises
        ------
        CustomException
            If there is an error during model evaluation.
        """
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            logger.info(f'Model evaluation metrics: {metrics}')
            return metrics
        except Exception as e:
            raise CustomException(e, sys)
    
    def log_model(self, metrics):
        """
        Logs the model, parameters, and metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            The evaluation metrics.

        Raises
        ------
        CustomException
            If there is an error during model logging.
        """
        try:
            mlflow.set_experiment('NHPA-Doctor-Visits')
            with mlflow.start_run():
                mlflow.log_params(self.model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(self.model, "model")
                logger.info('Model logged to MLflow successfully')
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_model(self):
        """
        Saves the trained model locally.

        Raises
        ------
        CustomException
            If there is an error during model saving.
        """
        try:
            model_path = os.path.join(self.local_model_path, 'random_forest_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f'Model saved locally at {model_path}')
        except Exception as e:
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    # Set the data path where the preprocessed data is saved
    data_path = os.path.join(project_root, 'data/processed/NHPA-doctor-visits-preprocessed-version1')
    local_model_path = os.path.join(project_root, 'models')

    model_creator = ModelCreation(data_path, local_model_path)
    model_creator.train_model()
    metrics = model_creator.evaluate_model()
    model_creator.log_model(metrics)
    model_creator.save_model()