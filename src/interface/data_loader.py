"""
Author: arcsi1989
"""
from abc import ABC, abstractmethod
from typing import Dict


class DataLoader(ABC):
    """Implements and interface for data loaders"""

    def __init__(self, config: Dict):
        self.config = config
        self.data = None

    @abstractmethod
    def load_data(self):
        """Loads data"""
        pass
