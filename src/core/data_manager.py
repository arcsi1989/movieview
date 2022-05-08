"""
Author: arcsi1989
"""
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from src.interface import DataManager


class MovieDataManager(DataManager):
    """
    Movie data manager responsible for loading, preprocessing, providing training and inference data
    The class beyond the base class contains the following attributes:
        _training_data(pd.DataFrame): Training data created during processing of the data
        _inference_data(pd.DataFrame): Any data without labels was assigned to this attribute
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): A configuration file containing the path to the file to be loaded.
        """
        super().__init__(config=config)
        self._path_to_file = self.config.get('path_to_file', None)
        self._training_data = None
        self._inference_data = None

    def load_data(self):
        """Loads CSV data from provided path (i.e. url) using pandas"""
        if self._path_to_file:
            self.data = pd.read_csv(self._path_to_file)
        else:
            ValueError(f"No path to file was provided")

    def process_data(self):
        """Processed the provided data"""
        known_views = {'Forrest Gump': 10000000,
                       'The Usual Suspects': 7500000,
                       'Rear Window': 6000000,
                       'North by Northwest': 4000000,
                       'The Secret in Their Eyes': 3000000,
                       'Spotlight': 1000000}

        self.data['view'] = self.data['Title'].map(known_views)
        self.data["Year"] = 2022 - self.data["Year"]
        self.data['Rating Count'] = self.data['Rating Count'].apply(lambda x: x.replace(",", "")).astype(int)

        limited_data = self.data[~(self.data['view'].isna())]

        self._training_data = limited_data.iloc[:, 2:]
        self._inference_data = self.data[(self.data['view'].isna())].iloc[:, 2:-1]
        print(self._inference_data)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the training data as a tuple - x and y"""
        # Define predictions and target
        predictors = ['Year', 'Rating', 'Rating Count']
        targets = ['view']

        x = self._training_data[predictors].values
        y = self._training_data[targets].values

        return x, y

    def get_inference_data(self) -> np.ndarray:
        """Gets the inference data from the provided dataset"""
        return self._inference_data.values
