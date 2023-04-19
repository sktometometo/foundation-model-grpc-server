#!/usr/bin/env python

import logging

import grpc

from LAVIS_grpc_server import lavis_server_pb2, lavis_server_pb2_grpc


def run():

  with grpc.insecure_channel('localhost:50051') as channel:
    stub = lavis_server_pb2_grpc.LAVISServerStub(channel)
    request = lavis_server_pb2.ImageCaptioningRequest()
    result = stub.ImageCaptioning(request)
    print('result: {}'.format(result))


if __name__ == '__main__':
  logging.basicConfig()
  run()
  print('Finished')
