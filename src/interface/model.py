"""
Author: arcsi1989
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Model(ABC):
    """
    This model interfaces defines the common methods to be exposed by a machine learning model.
    The interface defines the following methods for the models:
        - __init__: initialization / instantiation of a model architecture using a set of parameters
        - fit: training procedure of the model
        - predict: predict values based on input

    Furthermore, the model class exposes the following attributes:
        - _config: configuration dictionary used at model instantiation
    """
    def __init__(self, config: Dict):
        self._config = config

    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """ Fit the model to the input data"""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Predict the target values based on the prediction problem."""
        pass
