"""
Author: @arcsi1989
"""
import click
import os
import json

import numpy as np

from src.cli import src_cli
from src.core.sklearn_models import SklearnRegressionModel


def inference_exec(model: SklearnRegressionModel, data: np.ndarray, output_folder: str = None):
    """
    Executes the inference of the  data using the "provided model, and stores its outcome into the output_folder
    Args:
       model (SklearnRegressionModel): A trained sklearn model
       data (np.ndarray): A numpy array containing data for inference
       output_folder (str): A string defining the output folder
    """
    prediction = model.predict(data)

    if output_folder:
        result_path = f"{output_folder}/results"
        isExist = os.path.exists(result_path)
        if not isExist:
            os.makedirs(result_path)

        with open(f"{result_path}/inference.json", "w") as outfile:
            json.dump({'predictions': list(prediction)}, outfile)


@src_cli.command(help_priority=2)
@click.help_option("-h")
def inference():
    print('LOG | Pipeline for predicting movie views using a pretrained model')
    # TODO
    if False:
        raise NotImplementedError("Method is not implemented")
