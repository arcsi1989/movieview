# Program for the prediction of movie views

This program (1) downloads a list of movie files from a provided url defined by environment variable `DATA_URL`, (2) 
trains a regression model based on the provided sklearn model type `MODEL_TYPE` and model name `MODEL_NAME` also
provided as environmental variables.

## Quick start

The following procedure is tested with Python 3.8.0.

- Tested model_types: `linear_model` and `ensemble`;
- Tested models: `LinearRegression`, `RandomForestRegressor`

Optional: Create a virtual or conda environment and activate it.
1. Install movieview package
```shell
moviewview$ pip install -e .
```

2. Run the application
```shell
movieview$ export DATA_URL="https://raw.githubusercontent.com/WittmannF/imdb-tv-ratings/master/top-250-movie-ratings.csv"
movieview$ export MODEL_TYPE="linear_model"
movieview$ export MODEL_NAME="LinearRegression"
movieview$ task3 run-pipeline --output_dir <path_to_output_dir>
```

## Building and running a Docker image locally
This requires that you have installed Docker and the docker engine is running.

1. Build the Docker image
```shell
movieview$ docker build -it movies_view_predictor .
```

2. Run the Docker image
```shell
# url = "https://s3.amazonaws.com/products-42matters/test/biographies.list.gz"
# local_computer_path = Local computer path which will be mounted to the container to store the output
movieview$ docker run -e DATA_URL=<url> -v <local_computer_path>:/usr/src/data word_counter
```

Or even easier. Configure the environment variables in `docker-compose.yml` and run:
```shell
movieview$ docker-compose up
```
If the image `word_counter` has not been built yet, this will happen automatically.
