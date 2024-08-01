import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

# Directly set the project root directory
project_root = "/workspaces/real-end-to-end-ml"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
from src.logging.custom_logger import logger  # Import the custom logger
from src.custom_exceptions import CustomException  # Import the custom exception

class FeatureSelector:
    """
    A class used to perform feature engineering and selection on a dataset.

    Attributes
    ----------
    data : pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Methods
    -------
    load_data(path):
        Loads data from the specified file path into a Pandas DataFrame.
    
    calculate_bmi():
        Calculates BMI and adds it as a new column.

    create_age_group():
        Creates an Age_Group column based on the Age column.
    
    calculate_correlations():
        Calculates the correlation coefficients for numerical features.
    
    perform_chi_square():
        Performs the chi-square test for categorical features.
    
    select_features():
        Selects the important features based on correlation and chi-square test results.
    
    save_selected_features(file_path: str):
        Saves the selected features to a CSV file at the specified path.
    """
    
    def __init__(self, path):
        """
        Constructs all the necessary attributes for the FeatureSelector object.

        Parameters
        ----------
        path : str
            The path to the directory containing the data file.
        """
        try:
            self.data = self.load_data(path)
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

    def create_age_group(self):
        """
        Creates an Age_Group column based on the Age column.

        Raises
        ------
        CustomException
            If there is an error creating the Age_Group column.
        """
        try:
            logger.info('Creating Age_Group column')
            self.data['Age_Group'] = pd.cut(self.data['Age'], bins=[0, 18, 35, 50, 100], labels=['Teen', 'Young Adult', 'Adult', 'Senior'])
            logger.info('Age_Group column created')
        except Exception as e:
            raise CustomException(e, sys)

    def calculate_correlations(self):
        """
        Calculates the correlation coefficients for numerical features.

        Returns
        -------
        pd.Series
            The correlation coefficients with the target variable.

        Raises
        ------
        CustomException
            If there is an error calculating correlations.
        """
        try:
            logger.info('Calculating correlation coefficients for numerical features')
            le = LabelEncoder()
            self.data['NObeyesdad_encoded'] = le.fit_transform(self.data['NObeyesdad'])
            numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
            correlation_matrix = self.data[numerical_columns + ['NObeyesdad_encoded']].corr()
            correlation_with_target = correlation_matrix['NObeyesdad_encoded'].sort_values(ascending=False)
            logger.info('Correlation coefficients calculated')
            return correlation_with_target
        except Exception as e:
            raise CustomException(e, sys)

    def perform_chi_square(self):
        """
        Performs the chi-square test for categorical features.

        Returns
        -------
        pd.Series
            The chi-square scores for categorical features.
        pd.Series
            The p-values for the chi-square test for categorical features.

        Raises
        ------
        CustomException
            If there is an error performing the chi-square test.
        """
        try:
            logger.info('Performing chi-square test for categorical features')
            categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
            le = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = le.fit_transform(self.data[col])
            X_categorical = self.data[categorical_columns]
            y = self.data['NObeyesdad_encoded']
            chi_scores = chi2(X_categorical, y)
            chi2_scores = pd.Series(chi_scores[0], index=categorical_columns)
            chi2_p_values = pd.Series(chi_scores[1], index=categorical_columns)
            logger.info('Chi-square test performed')
            return chi2_scores, chi2_p_values
        except Exception as e:
            raise CustomException(e, sys)

    def select_features(self):
        """
        Selects the important features based on correlation and chi-square test results.

        Returns
        -------
        list
            The selected important features.

        Raises
        ------
        CustomException
            If there is an error selecting the features.
        """
        try:
            correlation_with_target = self.calculate_correlations()
            chi2_scores, chi2_p_values = self.perform_chi_square()

            selected_features = []

            # Selecting numerical features based on correlation
            selected_features.extend(correlation_with_target.index[1:5])  # Select top 4 numerical features excluding the target

            # Selecting categorical features based on chi-square test
            selected_features.extend(chi2_scores.sort_values(ascending=False).index)

            logger.info('Features selected based on correlation and chi-square test')
            return selected_features
        except Exception as e:
            raise CustomException(e, sys)

    def save_selected_features(self, selected_features, file_path: str):
        """
        Saves the selected features to a CSV file.

        Parameters
        ----------
        selected_features : list
            The list of selected features.
        file_path : str
            The path where the selected features CSV file will be saved.

        Raises
        ------
        CustomException
            If there is an error saving the selected features.
        """
        try:
            selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
            selected_features_df.to_csv(file_path, index=False)
            logger.info(f'Selected features saved successfully to {file_path}')
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Example usage
        data_path = "data/raw/cleaned_data.csv"
        output_path = "src/feature_selection/selected_features.csv"
        
        feature_selector = FeatureSelector(data_path)
        feature_selector.calculate_bmi()
        feature_selector.create_age_group()
        selected_features = feature_selector.select_features()
        feature_selector.save_selected_features(selected_features, output_path)
    except Exception as e:
        logger.error(f"Error in feature selection process: {e}")
        raise CustomException(e, sys)
