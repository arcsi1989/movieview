"""
Author: arcsi1989
"""
from typing import Tuple
import os

import numpy as np
import pandas as pd


def manage_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    A method to download and process data for training
    Returns:
        x(np.ndarray): A numpy array containing the predictor data
        y(np.ndarray): A numpy array containing the target data
    """
    url = os.getenv('DATA_URL')
    if url is None:
        raise ValueError("There is no 'DATA_URL environment variable defined")

    # Read data from url
    data = pd.read_csv(url)

    # TODO make it externally configurable
    known_views = {'Forrest Gump': 10000000,
                   'The Usual Suspects': 7500000,
                   'Rear Window': 6000000,
                   'North by Northwest': 4000000,
                   'The Secret in Their Eyes': 3000000,
                   'Spotlight': 1000000}

    data['view'] = data['Title'].map(known_views)
    data["Year"] = 2022 - data["Year"]
    data['Rating Count'] = data['Rating Count'].apply(lambda x: x.replace(",", "")).astype(int)
    limited_data = data[~(data['view'].isna())]

    # Define predictions and target
    predictors = ['Year', 'Rating', 'Rating Count']
    targets = ['view']

    x = limited_data[predictors].values
    y = limited_data[targets].values

    return x, y
