"""
Author: @arcsi1989
"""
import click
import os
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from src.cli import src_cli
from src.core import manage_data
from src.core.sklearn_models import SklearnRegressionModel



@src_cli.command(help_priority=2)
@click.help_option("-h")
def evaluate_model():
    print('LOG | Pipeline for evaluating a set of model for predicting moview views')

    # Retrieve model type and model name
    model_type = os.getenv('MODEL_TYPE')
    model_name = os.getenv('MODEL_NAME')

    kf = KFold(6)
    x, y = manage_data()
    model = SklearnRegressionModel(model_type=model_type, model_name=model_name).model
    scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
