import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception_handler import CustomException
from src.logger import logging
from src.components.data_validation import DataValidation,DataValidationConfig
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

def Numofproducts(n):
    if n==1:
        return 'One'
    elif n==2:
        return 'Two'
    elif n==3 :
        return 'Three'
    else:
        return 'Four'

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "dataset.csv")
    train_data_path: str = os.path.join("artifacts", "train_dataset.csv")
    test_data_path: str = os.path.join("artifacts", "test_dataset.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiate Data Ingestion.")
        try:
            dataset = pd.read_csv("data\credit-card-customer-churn.csv")
            # Create directories recursively
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            
            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=1)
            logging.info("Saving train and test data to artifacts.")
            dataset.drop(columns=['RowNumber','CustomerId'],inplace=True)
            dataset.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            train_data['NumOfProducts']=train_data['NumOfProducts'].apply(Numofproducts)
            test_data['NumOfProducts']=test_data['NumOfProducts'].apply(Numofproducts)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully.")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    di=DataIngestion()
    train_path,test_path=di.initiate_data_ingestion()
    print("Data Ingestion Successful.")
    dv=DataValidation()
    dv.initiate_data_validation(train_data_path=train_path,test_data_path=test_path)
    print("Data Validation successful.")

