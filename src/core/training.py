import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

coeff = list()
intercept = list()
available_indices = set(np.arange(6))

predictors = ['Year', 'Rating', 'Rating Count']
targets = ['view']

x = limited_data[predictors].values
y = limited_data[targets].values

coeff = list()
intercept = list()
available_indices = set(np.arange(6))

kf = KFold(n_splits=6)
for train_index, test_index in kf.split(training_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression(fit_intercept=True)

    model.fit(x_train, y_train)
    pred_y = model.predict(x_test)
    ref_y = y_test

    print("Mean squared error: %.2f" % mean_squared_error(ref_y, pred_y))
    # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2_score(ref_y, pred_y))

    # Reference vs prediction valies
    print(f"Reference {ref_y} vs Prediction{pred_y}")

    coeff.append(model.coef_)
    intercept.append(model.intercept_)

averaged_model = LinearRegression()
averaged_model.coef_ = np.mean(coeff, axis=0)
averaged_model.intercept_ = np.mean(intercept, axis=0)
