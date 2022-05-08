# Program for the prediction of movie views

This program (1) downloads a list of movie files from a provided url defined by environment variable `DATA_URL`, (2) 
trains a regression model based on the provided sklearn model type `MODEL_TYPE` and model name `MODEL_NAME` also
provided as environmental variables.

## Quick start

The following procedure is tested with Python 3.8.0. The developed machine learning pipeline supports all sklearn models,
however only the followings were tested: 

- Tested model_types: `linear_model` and `ensemble`;
- Tested models: `LinearRegression`, `RandomForestRegressor`


Optional: Create a virtual or conda environment and activate it.
1. Install movieview package
```shell
moviewview$ pip install -e .
```

2. Create the environmental variables. They are used to configure the machine learning pipeline (data source, model choice).
Nevertheless, the internal data-manager is tailored for the task-specific dataset in terms of expected columns and its data type.

```shell
movieview$ source config/code.env
```

3.Run the application using the provided CLI interface - type `task3 -h` or `task3 --help` for help. 
```shell
movieview$ task3 train-model --output_dir <path_to_output_dir>
```

The train-model pipeline also performs inference on the non-labelled data using the trained model.
The results of the inference is stored if an output path is provided under `predictions.json` file.

The CLI interface currently only enable the following commans:
- `train-model`: trains the model defined by the environmental variables

## Building and running a Docker image locally
This requires that you have installed Docker and the docker engine is running.

1. Build the Docker image
```shell
movieview$ docker build -t movies_view_predictor .
```

2. Run the Docker image
```shell
movieview$ docker run --env-file config/code.env movies_view_predictor
```

Or even easier. Configure the environment variables in `docker-compose.yml` and run:
```shell
movieview$ docker-compose up
```
If the image `movies_view_predictor` has not been built yet, this will happen automatically.


## Extra
1. A bash script is provided for (re)building the built docker container.
```shell
movieview$ source cicd/build_docker.sh build
```

2. Additional CLI idea initiation is in place for potential deployment of the machine learning model only for inference

## Current limitations
The developed model interface allow for configuration of the models using a config dictionary (e.g. loaded from 
a config yaml file), however, there is no config file loader in place yet.

