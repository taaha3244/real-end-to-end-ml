import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Directly set the project root directory
project_root = "/workspaces/real-end-to-end-ml"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
from src.logging.custom_logger import logger  # Import the custom logger
from src.custom_exceptions import CustomException  # Import the custom exception

class DataCleaner:
    """
    A class used to clean data loaded from a specified path.

    Attributes
    ----------
    path : str
        The path to the directory containing the data file.
    data : pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Methods
    -------
    load_data():
        Loads data from the specified file path into a Pandas DataFrame.
    
    remove_duplicates():
        Removes duplicate rows from the dataset.
    
    check_binary_columns(binary_columns):
        Ensures that the specified binary columns contain only 0 or 1.
    
    save_cleaned_data(file_path: str):
        Saves the cleaned dataset to a CSV file at the specified path.
    
    get_cleaned_data():
        Returns the cleaned dataset as a Pandas DataFrame.
    """
    
    def __init__(self, path):
        """
        Constructs all the necessary attributes for the DataCleaner object.

        Parameters
        ----------
        path : str
            The path to the directory containing the data file.
        """
        try:
            self.path = path
            self.data = self.load_data()
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_data(self):
        """
        Loads data from the specified file path into a Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded data as a Pandas DataFrame.

        Raises
        ------
        CustomException
            If there is an error loading the data.
        """
        try:
            # Construct the path to the data file
            data_file_path = os.path.join(self.path, 'data/raw/NPHA-doctor-visits_version1.csv')
            # Load the data into a DataFrame
            df = pd.read_csv(data_file_path)
            logger.info(f'Data loaded successfully from {data_file_path}')
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def remove_duplicates(self):
        """
        Removes duplicate rows from the dataset.

        Raises
        ------
        CustomException
            If there is an error removing duplicates.
        """
        try:
            logger.info('Checking for duplicates')
            initial_row_count = self.data.shape[0]
            self.data = self.data.drop_duplicates()
            final_row_count = self.data.shape[0]
            logger.info(f'Removed {initial_row_count - final_row_count} duplicate rows')
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_binary_columns(self, binary_columns):
        """
        Ensures that the specified binary columns contain only 0 or 1.

        Parameters
        ----------
        binary_columns : list of str
            A list of column names that should be binary.

        Raises
        ------
        ValueError
            If any of the specified columns contain non-binary values.
        
        CustomException
            If there is an error checking the binary columns.
        """
        try:
            logger.info('Checking binary columns')
            for col in binary_columns:
                if not set(self.data[col].unique()).issubset({0, 1}):
                    raise ValueError(f"Column {col} contains non-binary values")
            logger.info('Binary columns check passed')
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_cleaned_data(self, file_path: str):
        """
        Saves the cleaned dataset to a CSV file.

        Parameters
        ----------
        file_path : str
            The path where the cleaned CSV file will be saved.

        Raises
        ------
        CustomException
            If there is an error saving the cleaned data.
        """
        try:
            self.data.to_csv(file_path, index=False)
            logger.info(f'Cleaned data saved successfully to {file_path}')
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_cleaned_data(self):
        """
        Returns the cleaned dataset.

        Returns
        -------
        pd.DataFrame
            The cleaned data as a Pandas DataFrame.

        Raises
        ------
        CustomException
            If there is an error returning the cleaned data.
        """
        try:
            logger.info('Returning cleaned data')
            return self.data
        except Exception as e:
            raise CustomException(e, sys)


 # Data Preprocessor class for Scaling, OHC 


class DataPreprocessor:
    """
    A class to preprocess data for machine learning models.

    Attributes
    ----------
    data : pd.DataFrame
        The data to be preprocessed.
    X : pd.DataFrame
        The features of the dataset.
    y : pd.Series
        The target variable of the dataset.
    preprocessor : ColumnTransformer
        The preprocessing pipeline.

    Methods
    -------
    build_preprocessor():
        Builds the preprocessing pipeline.
    
    preprocess_data():
        Preprocesses the data and splits it into training and testing sets.
    
    save_preprocessed_data(X_train, X_test, y_train, y_test):
        Saves the preprocessed training and testing data to the specified path.
    
    get_preprocessed_data():
        Returns the preprocessed training and testing data.
    """

    def __init__(self, data, save_path):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be preprocessed.
        save_path : str
            The path where the preprocessed data will be saved.
        """
        self.data = data
        self.save_path = save_path
        self.X = self.data.drop(columns=['Number of Doctors Visited'])
        self.y = self.data['Number of Doctors Visited']
        self.categorical_features = ['Race', 'Gender']
        self.numerical_features = [col for col in self.X.columns if col not in self.categorical_features]
        self.preprocessor = self.build_preprocessor()
    
    def build_preprocessor(self):
        """
        Builds the preprocessing pipeline.

        Returns
        -------
        ColumnTransformer
            The preprocessing pipeline.
        """
        try:
            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, self.numerical_features),
                    ('cat', categorical_pipeline, self.categorical_features)
                ])
            
            logger.info('Preprocessing pipeline created successfully')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_data(self):
        """
        Preprocesses the data and splits it into training and testing sets.

        Returns
        -------
        tuple
            The preprocessed training and testing data.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            logger.info('Data preprocessed and split into training and testing sets successfully')
            return X_train_processed, X_test_processed, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test):
        """
        Saves the preprocessed training and testing data to the specified path.

        Parameters
        ----------
        X_train : array-like
            The preprocessed training features.
        
        X_test : array-like
            The preprocessed testing features.
        
        y_train : array-like
            The training target variable.
        
        y_test : array-like
            The testing target variable.

        Raises
        ------
        CustomException
            If there is an error saving the preprocessed data.
        """
        try:
            directory_path = os.path.join(self.save_path, 'NHPA-doctor-visits-preprocessed-version1')
            os.makedirs(directory_path, exist_ok=True)
            
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
            y_train_df = pd.DataFrame(y_train)
            y_test_df = pd.DataFrame(y_test)
            
            X_train_df.to_csv(os.path.join(directory_path, 'X_train.csv'), index=False)
            X_test_df.to_csv(os.path.join(directory_path, 'X_test.csv'), index=False)
            y_train_df.to_csv(os.path.join(directory_path, 'y_train.csv'), index=False)
            y_test_df.to_csv(os.path.join(directory_path, 'y_test.csv'), index=False)
            
            logger.info(f'Preprocessed data saved successfully to {directory_path}')
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_preprocessed_data(self):
        """
        Returns the preprocessed training and testing data.

        Returns
        -------
        tuple
            The preprocessed training and testing data.
        """
        X_train_processed, X_test_processed, y_train, y_test = self.preprocess_data()
        self.save_preprocessed_data(X_train_processed, X_test_processed, y_train, y_test)
        return X_train_processed, X_test_processed, y_train, y_test
    

# Example usage
if __name__ == "__main__":
    cleaner = DataCleaner(project_root)
    cleaner.remove_duplicates()
    cleaner.check_binary_columns(['Stress Keeps Patient from Sleeping', 'Medication Keeps Patient from Sleeping',
    'Pain Keeps Patient from Sleeping', 'Bathroom Needs Keeps Patient from Sleeping',
    'Uknown Keeps Patient from Sleeping'])
    cleaner.save_cleaned_data(os.path.join(project_root, 'data/raw/cleaned_data.csv'))
    cleaned_data = cleaner.get_cleaned_data()

    preprocessor = DataPreprocessor(cleaned_data, os.path.join(project_root, 'data/processed'))
    X_train_processed, X_test_processed, y_train, y_test = preprocessor.get_preprocessed_data()
    print(X_train_processed.shape, X_test_processed.shape)