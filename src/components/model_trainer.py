import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import joblib

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Load the preprocessor
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            preprocessor = joblib.load(preprocessor_path)
            logging.info("Loaded preprocessor from %s", preprocessor_path)

            # Create a pipeline with the preprocessor and Linear Regression
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', LinearRegression())
            ])

            # Train the pipeline
            logging.info("Training Linear Regression model")
            pipeline.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = pipeline.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            logging.info("R² score on test set: %s", r2_square)

            # Save the entire pipeline (preprocessor + model)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=pipeline
            )
            logging.info("Saved trained pipeline to %s", self.model_trainer_config.trained_model_file_path)

            return r2_square, "Linear Regression"

        except Exception as e:
            raise CustomException(e, sys)