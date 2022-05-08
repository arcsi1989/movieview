#!/bin/bash
#
# Build Docker from docker-compose yaml

# Print help instructions
if [[ $1 == "--help" || $1 == '-h' ]]; then
  echo "-------------- DOCKER BUILDER DOCUMENTATION ---------------"
  echo "Build the Docker container:"
  echo "   source cicd/build_docker.sh build"
fi

# Build the Docker container and run it
if [[ $1 == "build" ]]; then
  docker-compose -f ./docker-compose.yaml up --build --force-recreate
fi
