# Pytorch Lightning Hydra with Docker

This is a simple example of how to use Pytorch Lightning with Hydra and Docker.

## Requirements

- Docker
- Docker Compose
- Kaggle API Token
- Python 3.11
- Pytorch
- Pytorch Lightning
- Hydra
- timm
- numpy
- scikit-learn
- gdown

## Kaggle Dataset - Dog Breed Identification

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/khushikhushikhushi/dog-breed-image-dataset).

This dataset contains 10000+ images of dogs across 10 breeds. The breeds are as follows:

- Golden Retriver
- German Shepherd
- Labrador Retriever
- Beagle
- Bulldog
- Poodle
- Rottweiler
- Yorkshire Terrier
- Boxer
- Dachshund


## Docker

### What is Docker?

Docker is a containerization platform that allows you to package your application with its dependencies into a container. This ensures that your application runs the same way on any machine.

### DevContainer

A DevContainer is a Docker container that allows you to set up a development environment and tooling within a docker container and interact with it using VSCode.

```json
{
    "name": "Dog Breed Classifier Dev Container",
    "dockerComposeFile": "../docker-compose.yaml",
    "service": "app",
    "workspaceFolder": "/app",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python",
                "python.linting.enabled": true,
                "python.linting.pylintEnabled": true,
                "python.formatting.autopep8Path": "/usr/local/bin/autopep8",
                "python.formatting.blackPath": "/usr/local/bin/black",
                "python.formatting.yapfPath": "/usr/local/bin/yapf",
                "python.linting.banditPath": "/usr/local/bin/bandit",
                "python.linting.flake8Path": "/usr/local/bin/flake8",
                "python.linting.mypyPath": "/usr/local/bin/mypy",
                "python.linting.pycodestylePath": "/usr/local/bin/pycodestyle",
                "python.linting.pydocstylePath": "/usr/local/bin/pydocstyle"
            }
        }
    },
    "postCreateCommand": "pip install --user pylint autopep8 black yapf bandit flake8 mypy pycodestyle pydocstyle",
    "remoteUser": "root",
    "remoteEnv": {
        "KAGGLE_USERNAME": "${localEnv:KAGGLE_USERNAME}",
        "KAGGLE_KEY": "${localEnv:KAGGLE_KEY}"
    }
}
```

- `dockerComposeFile` is the path to the docker-compose file.
- `service` is the name of the service to run.
- `workspaceFolder` is the path to the workspace folder.
- `customizations` is the customizations to the VSCode settings.
- `remoteUser` is the remote user to use.
- `remoteEnv` is the remote environment variables to use. This is necessary to set the Kaggle credentials in order to download the dataset.


### How is Docker Compose used?

Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services. Then, with a single command, you create and start all the services from your configuration.

Template:

```yaml
services:
  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - .:/opt/mount
    environment:
      - KAGGLE_USERNAME
      - KAGGLE_KEY
    command: python src/train.py

  infer:
    build:
      context: .
      dockerfile: Dockerfile.infer
    volumes:
      - .:/opt/mount
      - ./predictions:/opt/mount/predictions
    environment:
      - KAGGLE_USERNAME
      - KAGGLE_KEY
    ports:
      - "8000:8000"
    command: >
      sh -c "python src/infer.py --checkpoint_path /opt/mount/logs/dog_breed_classification/checkpoints/best_model.ckpt
      --data_dir /opt/mount/data
      --save_dir /opt/mount/predictions
      && python -m http.server 8000 --directory /opt/mount/predictions"

  eval:
    build:
      context: .
      dockerfile: Dockerfile.eval
    volumes:
      - .:/opt/mount
    environment:
      - KAGGLE_USERNAME
      - KAGGLE_KEY
    command: python src/eval.py

volumes:
  data:
  predictions:
```

- `build` is the path to the Dockerfile to use.
- `volumes` is the volumes to mount.
- `environment` is the environment variables to set.
- `command` is the command to run.


### How to run the training container?

```bash
docker compose run train python src/train.py --max_epochs 5
```

To run the inference container, use the following command:

```bash
docker compose run infer python src/infer.py --checkpoint_path /opt/mount/checkpoints/epoch=29-step=10938.ckpt
```

To run the evaluation container, use the following command:

```bash
docker compose run eval python src/eval.py --ckpt_path /opt/mount/checkpoints/epoch=29-step=10938.ckpt
```

## Results

Evaluation results:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.9863013625144958     │
│         test_loss         │    0.0423024520277977     │
└───────────────────────────┴───────────────────────────┘
```

Sample predictions:

![Sample Predictions](https://github.com/aakashvardhan/lightning-hydra-tutorial/blob/main/predictions/test_predictions.png)