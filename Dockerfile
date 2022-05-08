FROM python:3

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -e .

CMD [ "task3", "train-model" ]
