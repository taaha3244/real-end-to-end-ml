import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import  LabelEncoder
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
            data_file_path = os.path.join(self.path, 'data/raw/ObesityDataset_raw.csv')
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
    A class used to preprocess data for ML classification tasks.

    Attributes
    ----------
    data : pd.DataFrame
        The loaded data as a Pandas DataFrame.
    selected_features : list
        The list of selected features.

    Methods
    -------
    load_data(path):
        Loads data from the specified file path into a Pandas DataFrame.
    
    load_selected_features(path):
        Loads selected features from the specified file path into a list.
    
    scale_numerical_features():
        Scales numerical features using StandardScaler.
    
    encode_categorical_features():
        Encodes categorical features using OneHotEncoder.
    
    encode_target():
        Encodes the target variable NObeyesdad using LabelEncoder.
    
    save_preprocessed_data(file_path: str):
        Saves the preprocessed data to a CSV file at the specified path.
    """
    
    def __init__(self, data_path, features_path):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters
        ----------
        data_path : str
            The path to the directory containing the data file.
        features_path : str
            The path to the file containing the selected features.
        """
        try:
            self.data = self.load_data(data_path)
            self.selected_features = self.load_selected_features(features_path)
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_data(self, path):
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
            # Load the data into a DataFrame
            df = pd.read_csv(path)
            logger.info(f'Data loaded successfully from {path}')
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_selected_features(self, path):
        """
        Loads selected features from the specified file path into a list.

        Returns
        -------
        list
            The list of selected features.

        Raises
        ------
        CustomException
            If there is an error loading the selected features.
        """
        try:
            # Load the selected features into a DataFrame
            selected_features_df = pd.read_csv(path)
            selected_features = selected_features_df['Selected_Features'].tolist()
            logger.info(f'Selected features loaded successfully from {path}')
            return selected_features
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_bmi(self):
        """
        Calculates BMI and adds it as a new column.

        Raises
        ------
        CustomException
            If there is an error calculating BMI.
        """
        try:
            logger.info('Calculating BMI')
            self.data['BMI'] = self.data['Weight'] / (self.data['Height'] ** 2)
            logger.info('BMI calculated and added as a new column')
        except Exception as e:
            raise CustomException(e, sys)    
    
    def scale_numerical_features(self):
        """
        Scales numerical features using StandardScaler.

        Raises
        ------
        CustomException
            If there is an error scaling the numerical features.
        """
        try:
            numerical_features = ['BMI', 'Weight', 'Age', 'CH2O']
            scaler = StandardScaler()
            self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
            logger.info('Numerical features scaled successfully')
        except Exception as e:
            raise CustomException(e, sys)
    
    def encode_categorical_features(self):
        """
        Encodes categorical features using OneHotEncoder.

        Raises
        ------
        CustomException
            If there is an error encoding the categorical features.
        """
        try:
            categorical_features = ['Gender', 'SCC', 'family_history_with_overweight', 'MTRANS', 'CAEC', 'SMOKE', 'FAVC', 'CALC']
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_categorical = encoder.fit_transform(self.data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)
            self.data = pd.concat([self.data, encoded_categorical_df], axis=1)
            self.data.drop(columns=categorical_features, inplace=True)
            logger.info('Categorical features encoded successfully')
        except Exception as e:
            raise CustomException(e, sys)
    
    def encode_target(self):
        """
        Encodes the target variable NObeyesdad using LabelEncoder.

        Raises
        ------
        CustomException
            If there is an error encoding the target variable.
        """
        try:
            label_encoder = LabelEncoder()
            self.data['NObeyesdad_encoded'] = label_encoder.fit_transform(self.data['NObeyesdad'])
            self.data.drop(columns=['NObeyesdad'], inplace=True)
            logger.info('Target variable encoded successfully')
        except Exception as e:
            raise CustomException(e, sys)

    def save_preprocessed_data(self, file_path: str):
        """
        Saves the preprocessed data to a CSV file.

        Parameters
        ----------
        file_path : str
            The path where the preprocessed data CSV file will be saved.

        Raises
        ------
        CustomException
            If there is an error saving the preprocessed data.
        """
        try:
            self.data.to_csv(file_path, index=False)
            logger.info(f'Preprocessed data saved successfully to {file_path}')
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Example usage
        data_path = "data/raw/cleaned_data.csv"
        features_path = "src/feature_selection/selected_features.csv"
        output_path = "data/processed/preprocessed_data.csv"
        
        preprocessor = DataPreprocessor(data_path, features_path)
        preprocessor.calculate_bmi()
        preprocessor.scale_numerical_features()
        preprocessor.encode_categorical_features()
        preprocessor.encode_target()
        preprocessor.save_preprocessed_data(output_path)
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise CustomException(e, sys)
    

# # Example usage
# if __name__ == "__main__":
#     cleaner = DataCleaner(project_root)
#     cleaner.remove_duplicates()
#     cleaner.save_cleaned_data(os.path.join(project_root, 'data/raw/cleaned_data.csv'))
#     cleaned_data = cleaner.get_cleaned_data()

