# LAVIS-grpc-server

This is a python package provides grpc server for [LAVIS](https://github.com/salesforce/LAVIS).

## Install

```bash
pip install -e .
```

You may have to install extra packages so that python modules can work.

## RUN

To run the server

```bash
run_lavis_server
```

You can check a client demo

```bash
sample_lavis_client
```

## Docker image

There is a docker image for server

```bash
docker built -t sktometometo/lavis-grpc-server .
```

and run the server.

```bash
docker run -p 50051:50051 --gpus all sktometometo/lavis-grpc-server
```
