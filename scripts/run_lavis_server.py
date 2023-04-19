import logging
from concurrent import futures

import grpc

from LAVIS_grpc_server.lavis_server_pb2_grpc import \
    add_LAVISServerServicer_to_server
from LAVIS_grpc_server.server import LAVISServer


def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  add_LAVISServerServicer_to_server(LAVISServer(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()


if __name__ == '__main__':
  logging.basicConfig()
  serve()
