"""
Author: @arcsi1989
"""
import os
import click

import numpy as np
from sklearn.model_selection import KFold
# TODO create interface for metrics and possiby a manager
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.cli import src_cli
from src.core import manage_data
from src.core.sklearn_models import SklearnRegressionModel


@src_cli.command(help_priority=0)
@click.help_option("-h")
@click.option("-o", "--output_folder",
              type=str,
              help="Provide a folder where the output should be placed")
def train_model(output_folder: str):
    """Training procedure of a SKlearn Regression model to predict movie views"""

    print('LOG | Pipeline for training a model for movie view prediction is initiated')

    print('LOG | Retrieve data from provided url and prorcess it for training')


    # Preprocess received data
    print('LOG | Preprocessing data')
    x, y = manage_data()

    print('LOG | Training model')
    # Retrieve model type and model name
    model_type = os.getenv('MODEL_TYPE')
    model_name = os.getenv('MODEL_NAME')

    if model_type == 'linear_model':
        coeff = list()
        intercept = list()

    scores = {'mse': list(), 'r2': list(), 'mae': list()}
    models = list()

    kf = KFold(n_splits=6)
    for train_index, test_index in kf.split(np.concatenate((x,y), axis=1)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SklearnRegressionModel(model_type=model_type, model_name=model_name)
        model.fit(x_train, y_train)
        pred_y = model.predict(x_test)
        ref_y = y_test

        # Storing the fold specific model and its performance evaluted on the test set
        models.append(model)
        scores['mse'].append(mean_squared_error(ref_y, pred_y))
        scores['mae'].append( mean_absolute_error(ref_y, pred_y))
        scores['r2'].append(r2_score(ref_y, pred_y))

        if model_type == 'linear_model':
            coeff.append(model.model.coef_)
            intercept.append(model.model.intercept_)

    print('LOG | Finished model training')

    # Creating and average  model 'coeff' 'and intercept' it was a linear model (e.g. LinearRegression)
    if model_type == 'linear_model':
        model = SklearnRegressionModel(model_type=model_type, model_name=model_name)
        model.model.coef_ = np.mean(coeff, axis=0)
        model.model.intercept_ = np.mean(intercept, axis=0)

    else:  # If the model is an ensemble model select best performing model based on the metrics
        voting = list()
        for metric in scores.keys():
            max_value = max(scores[metric])
            max_index = scores[metric].index(max_value)
            voting.append(max_index)
        model_idx = max(set(voting), key=voting.count)
        model = models[model_idx]

    # Evaluate model performance on all provided training data
    print('LOG | Model performance on all provided training data')
    pred_y = model.predict(x)
    ref_y = y
    print("Mean squared error: %.2f" % mean_squared_error(ref_y, pred_y))
    print("Mean absolute error: %.2f" % mean_absolute_error(ref_y, pred_y))
    print("Coefficient of determination: %.2f" % r2_score(ref_y, pred_y))
