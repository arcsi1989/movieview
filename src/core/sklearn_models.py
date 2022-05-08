"""
Author: arcsi1989
"""
from typing import Dict, Any
import importlib

import sklearn

from src.interface import Model


class SklearnRegressionModel(Model):
    """
    The Sklearn regression model interface wraps around scikit-learn models thus enabling instantiation of different
    sklearn regression models using the same interface.
    A model is initialized with its name and corresponding configuration if required.
    """
    def __init__(self, model_type: str, model_name: str, config: Dict = None):
        """
        Args:
            model_type (str): Type of the model (e.g. linear_model, ensemble)
            model_name (str): Name of the model as it defined within the Sklearn
            config (Dict): A dictionary containing the model configuration if required
        """
        if config is None:
            config = dict()
        super().__init__(config=config)

        self.model = getattr(importlib.import_module(f"sklearn.{model_type}"), model_name)(**config)

    def fit(self, *args, **kwargs) -> Any:
        """Performs the training/fitting of the model"""
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Any:
        """Perform the inference using the model"""
        return self.model.predict(*args, **kwargs)
