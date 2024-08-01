import os
import sys
# Directly set the project root directory
project_root = "/workspaces/real-end-to-end-ml"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from src.logging.custom_logger import logger  # Importing the custom logger
from src.custom_exceptions import CustomException  # Importing the custom exception


# Load environment variables
load_dotenv()

class DataIngestion:
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )
            logger.info("Initialized DataIngestion with AWS S3 client")
        except Exception as e:
            raise CustomException(e, sys)

    def load_data_from_s3(self, bucket_name, object_key, local_file_path):
        try:
            logger.info(f"Starting to load data from S3 bucket: {bucket_name}, key: {object_key}")

            # Get the object from S3
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            logger.info("Successfully retrieved object from S3")

            # Read the object's content into a DataFrame
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            logger.info("Successfully read CSV content into DataFrame")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            logger.info(f"Ensured directory exists for path: {local_file_path}")

            # Save the DataFrame to a CSV file in the specified local path
            df.to_csv(local_file_path, index=False)
            logger.info(f"Data successfully loaded from S3 and saved to {local_file_path}")
        except Exception as e:
            raise CustomException(e, sys)

# Configuration
bucket_name = 'taaha-aws'
object_key = 'ObesityDataSet_raw_and_data_sinthetic.csv'
local_file_path = 'data/raw/ObesityDataset_raw.csv'

# Data ingestion
data_ingestion = DataIngestion()
data_ingestion.load_data_from_s3(bucket_name, object_key, local_file_path)
