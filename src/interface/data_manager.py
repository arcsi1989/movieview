"""
Author: arcsi1989
"""
from abc import ABC, abstractmethod
from typing import Dict


class DataManager(ABC):
    """
    Implements and interface for data manager
    The interface defines the following methods for the models:
        - __init__: initialization / instantiation of DataManager
        - load_data: loads data
        - process_data: process_data

    Furthermore, the data manager class exposes the following attributes:
        - config: configuration dictionary used at instantiation
        - data: original data
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data = None
        self._training_data = None

    @abstractmethod
    def load_data(self):
        """Loads data"""
        pass

    @abstractmethod
    def process_data(self):
        """Processes data"""
        pass
