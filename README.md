# foundation-model-grpc-server

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Black formatting](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_black.yml/badge.svg)](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_black.yml)
[![Python Linting Check](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_linting.yml/badge.svg)](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_linting.yml)
[![ROS build workflow](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/catkin_build.yml/badge.svg)](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/catkin_build.yml)
[![Python package](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_package.yaml/badge.svg)](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/python_package.yaml)
[![Docker Image Build Workflow](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/docker_build.yml/badge.svg)](https://github.com/sktometometo/foundation-model-grpc-server/actions/workflows/docker_build.yml)

This is a python package provides grpc server for foundation model.
Currently models below are supported.

- [LAVIS](https://github.com/salesforce/LAVIS).
- [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

## Bare python module

### Install

```bash
pip install -e .
```

You may have to install extra packages so that python modules can work.

If you want to use LLaMA-Adapter, please install `llama_adapter_v2_multimodal` package in [this LLaMA-Adapter fork](https://github.com/sktometometo/LLaMA-Adapter).

### RUN

To run the server

```bash
run_lavis_server
```

You can check a client demo

```bash
sample_lavis_client
```

### Docker image

There is a docker image for server

```bash
docker built -t sktometometo/foundation-model-grpc-server .
```

and run the server.

```bash
docker run -p 50051:50051 --gpus all sktometometo/foundation-model-grpc-server
```

## ROS Package

### Install

```bash
rosdep install --from-paths . --ignore-src -y -r
catkin build foundation_model_grpc_server
```

### How to use demo

After launch a server.

```bash
roslaunch foundation_model_grpc_server demo.launch
```
