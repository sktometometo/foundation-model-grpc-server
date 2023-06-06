# foundation-model-grpc-server

This is a python package provides grpc server for foundation model.
Currently models below are supported.

- [LAVIS](https://github.com/salesforce/LAVIS).

## Bare python module

### Install

```bash
pip install -e .
```

You may have to install extra packages so that python modules can work.

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
