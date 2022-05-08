"""
Author: arcsi1989
"""
from typing import Dict

import pandas as pd

from src.interface import DataLoader


class CSVLoader(DataLoader):
    """
    CSV data loader class:
    """
    def __init__(self, config: Dict):
        super().__init__(config=config)

    def load_data(self):
        """Loads CSV data using pandas"""
        self.data = pd.read_csv(self.config['path_to_file'])
