import sys
from pathlib import Path
import os

import yaml

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


def evaluate_model(x: np.ndarray, y: np.ndarray, cv: KFold):
    model = get_model('SVM')
    scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))


def get_model(model_name, random_state: int = 42):
    max_iter = 10000
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'Ridge':
        model = Ridge(random_state=random_state, max_iter=max_iter)
    elif model_name == 'Lasso':
        model = Lasso(random_state=random_state, max_iter=max_iter)
    elif model_name == 'ElasticNet':
        model = ElasticNet(random_state=random_state, max_iter=max_iter)
    elif model_name == 'SVM':
        model = svm.SVR()
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(random_state=random_state, criterion='squared_error')  # mse in older versions
    elif model_name == 'HistGradientBoostingRegressor':
        model = HistGradientBoostingRegressor(random_state=random_state,
                                              loss='squared_error')  # least_squared in older version
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror')
    return model


def pipeline():
    print('LOG | Pipeline is initiated')

    # config_file = Path(__file__).parent / "configs.yaml"
    # with open(config_file) as file:
    #    configs = yaml.full_load(file)

    url = os.getenv('DATA_URL')
    data = pd.read_csv(url)

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

    training_data = limited_data.iloc[:, 2:]

    predictors = ['Year', 'Rating', 'Rating Count']
    targets = ['view']

    x = limited_data[predictors].values
    y = limited_data[targets].values

    kf = KFold(n_splits=6)
    coeff = list()
    intercept = list()
    m_a_e = list()
    for train_index, test_index in kf.split(training_data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = get_model(model_name='LinearRegression')
        model.fit(x_train, y_train)
        pred_y = model.predict(x_test)
        ref_y = y_test

        # The mean squared error
        # print("Mean squared error: %.2f" % mean_squared_error(ref_y, pred_y))
        # The coefficient of determination: 1 is perfect prediction
        # print("Coefficient of determination: %.2f" % r2_score(ref_y, pred_y))

        # Reference vs prediction valies
        # print(f"Reference {ref_y} vs Prediction{pred_y}")
        # print("Mean absolute error: %.2f" % mean_absolute_error(ref_y, pred_y))
        #pred_y = model.predict(x)
        #ref_y = y
        #print("Mean squared error: %.2f" % mean_squared_error(ref_y, pred_y))
        #print("Mean absolute error: %.2f" % mean_absolute_error(ref_y, pred_y))
        #print("Coefficient of determination: %.2f" % r2_score(ref_y, pred_y))

        coeff.append(model.coef_)
        intercept.append(model.intercept_)
        m_a_e.append(mean_absolute_error(ref_y, pred_y))

    own_model = get_model(model_name='LinearRegression')
    own_model.coef_ = np.mean(coeff, axis=0)
    own_model.intercept_ = np.mean(intercept, axis=0)
    pred_y = own_model.predict(x)
    ref_y = y
    print("Mean squared error: %.2f" % mean_squared_error(ref_y, pred_y))
    print("Mean absolute error: %.2f" % mean_absolute_error(ref_y, pred_y))
    print("Coefficient of determination: %.2f" % r2_score(ref_y, pred_y))


if __name__ == '__main__':
    pipeline()
